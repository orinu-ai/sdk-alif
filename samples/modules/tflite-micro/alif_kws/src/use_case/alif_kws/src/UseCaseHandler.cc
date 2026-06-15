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
#include <zephyr/console/console.h>
#include <string.h>

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
#define KW_WAKE_TH       0.75f   /* orinu (idx 0) */
#define KW_CMD_TH        0.60f   /* general commands */
#define KW_CRIT_TH       0.90f   /* emergency(23)/shutdown(39)/reset(63) */
#define KW_VAD_RMS       262.0f  /* int16: 0.008 * 32768 ~= 262 */
#define KW_COOLDOWN_MS   1500
/* 발화 이벤트 검출 상태 */
static int64_t kws_last_send = 0;             /* 전역 쿨다운 (A32 검증 방식) */
static float kws_cur_rms = 0.0f;

/* 빈도 기반(majority vote): 발화 구간 라벨별 횟수 + score합(동점 처리용) */
static uint32_t kws_vote_cnt[64]   = {0};
static float    kws_vote_score[64] = {0.0f};
static bool     kws_in_utterance   = false;

/* VAD 트리거 정렬: 발화 시작(STRIDE RMS 엣지) 감지 후 N윈도우만 추론 */
static bool     kws_prev_silent     = true;
static int      kws_infer_remaining = 0;
static int      kws_onset_delay     = 0;   /* 발화 감지 후 정렬 지연 윈도우 */
#define KW_INFER_WINDOWS  1   /* 정렬된 1윈도우만 추론 */
#define KW_ONSET_RMS      700.0f
#define KW_SCAN_CHUNK     800        /* 정밀 스캔 청크 (0.05s @16k) */
#define KW_ALIGN_GUARD    1600       /* 단어 앞 여백 (0.1s, 학습 정렬과 일치) */
static bool     kws_do_align        = false;  /* 이번 루프에서 정렬 추론할지 */   /* 발화 시작 판정용 (잡음 263~314 차단, 발화 1010+) */   /* 발화 시작 후 추론 윈도우 수 (0.5s x 2 = 1s) */

static float kw_threshold(uint32_t kwid)
{
	if (kwid == 0u) {
		return KW_WAKE_TH;
	}
	if (kwid == 21u) {  /* emergency */
                return KW_CRIT_TH;
        }
	return KW_CMD_TH;
}

/* eyeCAM: 프린터 전용 키워드 제외 (임시 블랙리스트, 최종 keyword set은 PM 확정 예정) */
static bool kw_is_printer_only(uint32_t kwid)
{
        (void)kwid;
        return false;  /* eyeCAM: 프린터 전용 키워드 없음 (모든 키워드 허용) */
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

/* DEBUG(MHU 무결성 검증용, 운영 배포 시 제거 가능): HE 송신 누적 카운터.
 * wire에 패킹되어 HP가 HE_sent vs HP_rcvd를 비교 가능. */
static uint32_t kws_he_sent = 0u;
static void kws_send_to_hp(uint32_t kw_id)
{
	if (kws_mhu0_s == NULL) {
		return;
	}
	kws_he_sent++;
	/* wire: [31:8]=sent counter, [7:0]=kw_id+1 */
	uint32_t msg = ((kws_he_sent & 0xFFFFFFu) << 8) | ((kw_id + 1u) & 0xFFu);
	ipm_send(kws_mhu0_s, 0, 0, &msg, sizeof(msg));
	LOG_INF("KWS->HP sent kw_id=%u (sent#%u)", kw_id, kws_he_sent);
}

#if defined(CONFIG_KWS_MEASURE_MODE)
/* 측정 모드 전송: top-1 label_idx + score(0~100) + window index를 wire에 패킹.
 * wire: [31:24]=score(0~100), [23:8]=window(하위16bit), [7:0]=label_idx+1 (none=0). */
static void kws_send_meas_to_hp(uint32_t label_idx, float score, uint32_t window)
{
	if (kws_mhu0_s == NULL) {
		return;
	}
	uint32_t sc100 = (uint32_t)(score * 100.0f + 0.5f);
	if (sc100 > 100u) sc100 = 100u;
	uint32_t lo = (label_idx == 0xFFu) ? 0u : ((label_idx + 1u) & 0xFFu);
	uint32_t msg = ((sc100 & 0xFFu) << 24) | ((window & 0xFFFFu) << 8) | lo;
	ipm_send(kws_mhu0_s, 0, 0, &msg, sizeof(msg));
}
#endif
#if defined(CONFIG_KWS_RECORD_MODE)
/* 녹음 모드: console "rec" → 1초 캡처 → UART 덤프 (PC가 wav 저장).
 * PC→HE: "rec"=녹음, "quit"=종료
 * HE→PC: "<REC_START len=N>" + N개 int16(공백) + "<REC_END>" */
#if defined(CONFIG_KWS_RECORD_MODE)
static int16_t rec_buf2s[AUDIO_SAMPLES * 2];  /* 2초 녹음 버퍼 */
#endif 
bool RecordAudioHandler(ApplicationContext &ctx)
{
        (void)ctx;
        const int REC_TOTAL = AUDIO_SAMPLES * 2;  /* 2초 = 32000 */
        printk("<REC_READY>\n");
        while (true) {
                char *line = console_getline();
                if (line == nullptr) { continue; }
                if (strncmp(line, "quit", 4) == 0) { printk("<REC_QUIT>\n"); return true; }
                if (strncmp(line, "rec", 3) != 0) { continue; }

                /* rec마다 마이크 새로 시작 */
                int aerr = audio_init(16000);
                if (aerr) { printk("<REC_INIT_ERR %d>\n", aerr); continue; }

                /* 2초 캡처: 1초 버전과 동일 패턴(get→wait→copy), 4회 */
                bool rec_ok = true;
                for (int filled = 0; filled < REC_TOTAL; filled += AUDIO_STRIDE) {
                        get_audio_data(audio_inf + AUDIO_SAMPLES, AUDIO_STRIDE);
                        if (wait_for_audio()) { rec_ok = false; break; }
                        std::copy(audio_inf + AUDIO_SAMPLES,
                                  audio_inf + AUDIO_SAMPLES + AUDIO_STRIDE,
                                  rec_buf2s + filled);
                }

                audio_uninit();  /* 마이크 정지 */
                if (!rec_ok) { printk("<REC_ERR>\n"); continue; }

                /* 2초 덤프 */
                float rms = kws_window_rms(rec_buf2s, REC_TOTAL);
                printk("<REC_RMS %d>\n", (int)rms);
                printk("<REC_START len=%d>\n", REC_TOTAL);
                for (int i = 0; i < REC_TOTAL; i++) {
                        printk("%d ", rec_buf2s[i]);
                        if ((i & 0x1FF) == 0x1FF) { k_yield(); }
                }
                printk("\n<REC_END>\n");
        }
        return true;
}
#endif
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

	// 부팅 직후 윈도우(audio_inf의 AUDIO_SAMPLES 본체)를 실제 오디오로 미리 채운다.
	// 안 그러면 앞부분이 0(미충전)이라 첫 ~2초간 추론 입력이 불완전 → 초기 인식 지연.
	// STRIDE 단위로 윈도우가 완전히 찰 때까지 선충전 (기존 슬라이드 패턴과 동일).
	for (int filled = 0; filled < AUDIO_SAMPLES; filled += AUDIO_STRIDE) {
		get_audio_data(audio_inf + AUDIO_SAMPLES, AUDIO_STRIDE);
		if (wait_for_audio()) {
			LOG_ERR("hal_get_audio_data failed during prefill");
			return false;
		}
		std::copy(audio_inf + AUDIO_STRIDE, audio_inf + AUDIO_STRIDE + AUDIO_SAMPLES,
		          audio_inf);
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

		/* ★ VAD 트리거 정렬: STRIDE 부분 RMS로 발화 시작(침묵->소리) 감지 */
		float kws_stride_rms = kws_window_rms(audio_inf + AUDIO_SAMPLES - AUDIO_STRIDE,
		                                      AUDIO_STRIDE);
		bool kws_now_silent = (kws_stride_rms < KW_ONSET_RMS);
		if (kws_prev_silent && !kws_now_silent) {
			/* 발화 시작 감지 → 1윈도우 지연(첫 불완전 윈도우 스킵) 후 N윈도우 추론 */
			printk("[vad] onset rms=%d\n", (int)kws_stride_rms);
			kws_onset_delay = 2;   /* 2루프 지연: 단어가 윈도우 앞쪽으로 오도록 */
		}
		kws_prev_silent = kws_now_silent;
		kws_do_align = false;
		if (kws_onset_delay > 0) {
			kws_onset_delay--;
			if (kws_onset_delay == 0) {
				kws_infer_remaining = KW_INFER_WINDOWS;
				kws_do_align = true;  /* 이번 루프: 정밀 정렬 추론 */
			}
		}

		/* ★ 정밀 정렬: 발화 시작 샘플 스캔 후 inferenceWindow 시프트 */
		if (kws_do_align) {
			int kws_p0 = -1;
			for (int s = 0; s + KW_SCAN_CHUNK <= AUDIO_SAMPLES; s += KW_SCAN_CHUNK) {
				if (kws_window_rms(audio_inf + s, KW_SCAN_CHUNK) >= KW_ONSET_RMS) {
					kws_p0 = s;
					break;
				}
			}
			int kws_shift = 0;
			if (kws_p0 >= 0) {
				kws_shift = kws_p0 - KW_ALIGN_GUARD;
				if (kws_shift < 0) kws_shift = 0;
				if (kws_shift > AUDIO_STRIDE) kws_shift = AUDIO_STRIDE;  /* 버퍼 안전: shift+SAMPLES<=SAMPLES+STRIDE */
			}
			inferenceWindow = audio_inf + kws_shift;
			printk("[align] p0=%d shift=%d\n", kws_p0, kws_shift);
		}

		/* 발화 구간(정렬된 N윈도우)에서만 추론 + 전송 */
		if (kws_infer_remaining > 0) {
			kws_infer_remaining--;

			uint32_t start = k_cycle_get_32();
			if (!preProcess.DoPreProcess(inferenceWindow, kws_do_align ? 0 : index)) {
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

#if defined(CONFIG_KWS_MEASURE_MODE)
			{
				uint32_t kwid_m = singleInfResult.empty() ? 0xFFu
				                  : (singleInfResult[0].m_labelIdx & 0xFFu);
				float sc_m = singleInfResult.empty() ? 0.0f
				             : singleInfResult[0].m_normalisedVal;
				kws_send_meas_to_hp(kwid_m, sc_m, (uint32_t)index);
			}
#else
			/* VAD 정렬 + 빈도: N윈도우 동안 득표 누적, 마지막 윈도우에서 최다 득표 전송 */
			if (!singleInfResult.empty()) {
				uint32_t kwid = singleInfResult[0].m_labelIdx;
				float sc = singleInfResult[0].m_normalisedVal;
				bool is_silence = (kwid == 30u || kwid == 31u);
				bool is_valid_kw = (kwid < 64u && !is_silence &&
				                    !kw_is_printer_only(kwid) && sc >= kw_threshold(kwid));
				if (is_valid_kw) {
					kws_vote_cnt[kwid]++;
					kws_vote_score[kwid] += sc;
				}
			}
			if (kws_infer_remaining == 0) {  /* N윈도우 끝 = 발화 끝 */
				uint32_t best_kwid = 0xFFFFFFFFu;
				uint32_t best_cnt = 0;
				float best_scoresum = 0.0f;
				for (uint32_t i = 0; i < 64u; i++) {
					if (kws_vote_cnt[i] > best_cnt ||
					    (kws_vote_cnt[i] == best_cnt && kws_vote_score[i] > best_scoresum)) {
						best_cnt = kws_vote_cnt[i];
						best_scoresum = kws_vote_score[i];
						best_kwid = i;
					}
				}
				if (best_kwid != 0xFFFFFFFFu) {
					printk("[gate] SENT vote kwid=%u cnt=%u scoresum=%d\n",
					       (unsigned)best_kwid, (unsigned)best_cnt,
					       (int)(best_scoresum * 100));
					kws_send_to_hp(best_kwid);
				}
				for (uint32_t i = 0; i < 64u; i++) {
					kws_vote_cnt[i] = 0; kws_vote_score[i] = 0.0f;
				}
			}
#endif
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
		}  /* end if (kws_infer_remaining > 0) -- VAD 게이트 */

		++index;
	} while (!oneshot);

	audio_uninit();

	return true;
}

static bool PresentInferenceResult(const std::vector<kws::KwsResult> &results)
{
	LOG_INF("Final results:");
	LOG_INF("Total number of inferences: %zu", results.size());


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
