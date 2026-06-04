/*
 * Copyright (c) 2021-2022 Arm Limited. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#include "UseCaseHandler.hpp"
#include "AudioBackend.hpp"
#include "KwsClassifier.hpp"
#include "MicroNetKwsModel.hpp"
#include "AudioUtils.hpp"
#include "KwsResult.hpp"
#include "KwsProcessing.hpp"

#include <vector>
#include <zephyr/kernel.h>
#include <zephyr/logging/log.h>
#include <zephyr/drivers/ipm.h>

LOG_MODULE_REGISTER(UseCaseHandler);

using arm::app::ApplicationContext;
using arm::app::ClassificationResult;
using arm::app::KwsClassifier;
using arm::app::KwsPostProcess;
using arm::app::KwsPreProcess;
using arm::app::MicroNetKwsModel;
using arm::app::Model;

#define AUDIO_SAMPLES  CONFIG_AUDIO_SAMPLES
#define AUDIO_STRIDE   CONFIG_AUDIO_STRIDE
#define RESULTS_MEMORY CONFIG_RESULTS_MEMORY

static int16_t audio_inf[AUDIO_SAMPLES + AUDIO_STRIDE];

namespace alif
{
namespace app
{

namespace audio
{
using namespace arm::app::audio;
}

namespace kws
{
using namespace arm::app::kws;
}

/**
 * @brief           Presents KWS inference results.
 * @param[in]       results     Vector of KWS classification results to be displayed.
 * @return          true if successful, false otherwise.
 **/
static bool PresentInferenceResult(const std::vector<arm::app::kws::KwsResult> &results);

/* MHU0 (HE->HP): send recognized KWS keyword id to HP core */
static const struct device *kws_mhu0_s = NULL;

/* KWS post-processing (per KWS-team README) */
#define KW_WAKE_TH       0.50f   /* orinu (idx 0) */
#define KW_CMD_TH        0.60f   /* general commands */
#define KW_CRIT_TH       0.90f   /* emergency(23)/shutdown(39)/reset(63) */
#define KW_VAD_RMS       262.0f  /* int16: 0.008 * 32768 ~= 262 */
#define KW_COOLDOWN_MS   1500
static int64_t kws_last_time[66] = {0};
static float kws_cur_rms = 0.0f;

static float kw_threshold(uint32_t kwid)
{
	if (kwid == 0u) {
		return KW_WAKE_TH;
	}
	if (kwid == 23u || kwid == 39u || kwid == 63u) {
		return KW_CRIT_TH;
	}
	return KW_CMD_TH;
}

static float kws_window_rms(const int16_t *buf, int n)
{
	double acc = 0.0;
	for (int i = 0; i < n; ++i) {
		double v = (double)buf[i];
		acc += v * v;
	}
	return (float)sqrt(acc / (double)n);
}

static void kws_mhu_init(void)
{
	kws_mhu0_s = DEVICE_DT_GET(DT_NODELABEL(rtsshe_rtsshp_mhu0_s));
	if (!device_is_ready(kws_mhu0_s)) {
		LOG_ERR("MHU0 (HE->HP) not ready");
		kws_mhu0_s = NULL;
	} else {
		LOG_INF("MHU0 (HE->HP) ready");
	}
}

static void kws_send_to_hp(uint32_t kw_id)
{
	if (kws_mhu0_s == NULL) {
		return;
	}
	uint32_t msg = kw_id + 1u; /* +1 so id 0 (orinu) sets a non-zero MHU doorbell bit */
	ipm_send(kws_mhu0_s, 0, 0, &msg, sizeof(msg));
	LOG_INF("KWS->HP sent kw_id=%u (wire=%u)", kw_id, msg);
}

/* KWS inference handler. */
bool ClassifyAudioHandler(ApplicationContext &ctx, bool oneshot)
{
	auto &model = ctx.Get<Model &>("model");
	const auto mfccFrameLength = ctx.Get<int>("frameLength");
	const auto mfccFrameStride = ctx.Get<int>("frameStride");
	const auto audioRate = ctx.Get<int>("audioRate");
	const auto scoreThreshold = ctx.Get<float>("scoreThreshold");

	constexpr int minTensorDims = static_cast<int>(
		(MicroNetKwsModel::ms_inputRowsIdx > MicroNetKwsModel::ms_inputColsIdx)
			? MicroNetKwsModel::ms_inputRowsIdx
			: MicroNetKwsModel::ms_inputColsIdx);

	if (!model.IsInited()) {
		LOG_ERR("Model is not initialised! Terminating processing.");
		return false;
	}

	/* Get Input and Output tensors for pre/post processing. */
	TfLiteTensor *inputTensor = model.GetInputTensor(0);
	TfLiteTensor *outputTensor = model.GetOutputTensor(0);
	if (!inputTensor->dims) {
		LOG_ERR("Invalid input tensor dims");
		return false;
	} else if (inputTensor->dims->size < minTensorDims) {
		LOG_ERR("Input tensor dimension should be >= %d", minTensorDims);
		return false;
	}

	/* Get input shape for feature extraction. */
	TfLiteIntArray *inputShape = model.GetInputShape(0);
	const uint32_t numMfccFeatures = inputShape->data[MicroNetKwsModel::ms_inputColsIdx];
	const uint32_t numMfccFrames =
		inputShape->data[arm::app::MicroNetKwsModel::ms_inputRowsIdx];

	/* We expect to be sampling 1 second worth of data at a time.
	 *  NOTE: This is only used for time stamp calculation. */
	const float secondsPerSample = 1.0f / audioRate;

	/* Set up pre and post-processing. */
	KwsPreProcess preProcess = KwsPreProcess(inputTensor, numMfccFeatures, numMfccFrames,
						 mfccFrameLength, mfccFrameStride);

	std::vector<ClassificationResult> singleInfResult;
	KwsPostProcess postProcess =
		KwsPostProcess(outputTensor, ctx.Get<KwsClassifier &>("classifier"),
			       ctx.Get<std::vector<std::string> &>("labels"), singleInfResult);

	int index = 0;
	std::vector<kws::KwsResult> infResults;
	int err = audio_init(audioRate);
	if (err) {
		LOG_ERR("hal_audio_init failed with error: %d", err);
		return false;
	}

	/* Init HE->HP MHU once (after audio is up) */
	static bool s_mhu_inited = false;
	if (!s_mhu_inited) {
		kws_mhu_init();
		s_mhu_inited = true;
	}

	// Start first fill of final stride section of buffer
	get_audio_data(audio_inf + AUDIO_SAMPLES, AUDIO_STRIDE);

	do {
		// Wait until stride buffer is full - initiated above or by previous interation of
		// loop
		int err = wait_for_audio();
		if (err) {
			LOG_ERR("hal_get_audio_data failed with error: %d", err);
			return false;
		}

		// move buffer down by one stride, clearing space at the end for the next stride
		std::copy(audio_inf + AUDIO_STRIDE, audio_inf + AUDIO_STRIDE + AUDIO_SAMPLES,
			  audio_inf);

		// start receiving the next stride immediately before we start heavy processing, so
		// as not to lose anything
		get_audio_data(audio_inf + AUDIO_SAMPLES, AUDIO_STRIDE);

		audio_preprocessing(audio_inf + AUDIO_SAMPLES - AUDIO_STRIDE, AUDIO_STRIDE);

		const int16_t *inferenceWindow = audio_inf;
		kws_cur_rms = kws_window_rms(inferenceWindow, AUDIO_SAMPLES);

		uint32_t start = k_cycle_get_32();
		/* Run the pre-processing, inference and post-processing. */
		if (!preProcess.DoPreProcess(inferenceWindow, index)) {
			LOG_ERR("Pre-processing failed.");
			return false;
		}
		LOG_INF("Preprocessing time = %.3f ms",
		       (double)(k_cycle_get_32() - start) / sys_clock_hw_cycles_per_sec() * 1000);

		start = k_cycle_get_32();
		if (!model.RunInference()) {
			LOG_ERR("Inference failed.");
			return false;
		}
		LOG_INF("Inference time = %.3f ms",
		       (double)(k_cycle_get_32() - start) / sys_clock_hw_cycles_per_sec() * 1000);

		start = k_cycle_get_32();
		if (!postProcess.DoPostProcess()) {
			LOG_ERR("Post-processing failed.");
			return false;
		}
		LOG_INF("Postprocessing time = %.3f ms",
		       (double)(k_cycle_get_32() - start) / sys_clock_hw_cycles_per_sec() * 1000);

		/* Add results from this window to our final results vector. */
		if (infResults.size() == RESULTS_MEMORY) {
			infResults.erase(infResults.begin());
		}
		infResults.emplace_back(kws::KwsResult(
			singleInfResult, index * secondsPerSample * preProcess.m_audioDataStride,
			index, scoreThreshold));

#if VERIFY_TEST_OUTPUT
		DumpTensor(outputTensor);
#endif /* VERIFY_TEST_OUTPUT */

		if (!PresentInferenceResult(infResults)) {
			return false;
		}

		++index;
	} while (!oneshot);

	audio_uninit();

	return true;
}

static bool PresentInferenceResult(const std::vector<kws::KwsResult> &results)
{
	LOG_INF("Final results:");
	LOG_INF("Total number of inferences: %zu", results.size());

	/* KWS post-processing per KWS-team README:
	 * VAD (window RMS) + per-keyword threshold + per-keyword cooldown.
	 * Uses the newest inference frame. */
	if (!results.empty() && !results.back().m_resultVec.empty()) {
		uint32_t kwid = results.back().m_resultVec[0].m_labelIdx;
		float sc = results.back().m_resultVec[0].m_normalisedVal;
		int64_t now = k_uptime_get();

		if (kws_cur_rms >= KW_VAD_RMS &&     /* 1. VAD: window loud enough */
		    kwid < 64u &&                    /* 2. ignore _silence_/_unknown_ */
		    sc >= kw_threshold(kwid)) {      /* 3. per-keyword threshold */
			if (now - kws_last_time[kwid] >= KW_COOLDOWN_MS) { /* 4. cooldown */
				kws_send_to_hp(kwid);
				kws_last_time[kwid] = now;
			}
		}
	}

	for (const auto &result : results) {

		std::string topKeyword{"<none>"};
		float score = 0.f;
		if (!result.m_resultVec.empty()) {
			topKeyword = result.m_resultVec[0].m_label;
			score = result.m_resultVec[0].m_normalisedVal;
		}

		if (result.m_resultVec.empty()) {
			LOG_INF("For timestamp: %f (inference #: %" PRIu32
			     "); label: %s; threshold: %f",
			     (double)result.m_timeStamp, result.m_inferenceNumber,
			     topKeyword.c_str(), (double)result.m_threshold);
		} else {
			for (uint32_t j = 0; j < result.m_resultVec.size(); ++j) {
				LOG_INF("For timestamp: %f (inference #: %" PRIu32
				     "); label: %s, score: %f; threshold: %f",
				     (double)result.m_timeStamp, result.m_inferenceNumber,
				     result.m_resultVec[j].m_label.c_str(),
				     result.m_resultVec[j].m_normalisedVal,
				     (double)result.m_threshold);
			}
		}
	}

	return true;
}

} /* namespace app */
} /* namespace alif */
