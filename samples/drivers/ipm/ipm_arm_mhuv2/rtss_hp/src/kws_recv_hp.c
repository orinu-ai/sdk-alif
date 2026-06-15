/* HP-side KWS receiver: receive kw_id from HE via MHU0, log it with label.
 * (Placeholder for future camera trigger on 'orinu')
 */
#include <zephyr/kernel.h>
#include <zephyr/sys/printk.h>
#include <zephyr/drivers/ipm.h>

const struct device *mhu0_r;

/* Keyword labels, index = kw_id. Must match HE Labels.cc order. */
static const char *kw_labels[32] = {
    "orinu", "eyecam", "capture", "photo",
    "record", "zoom_in", "zoom_out", "read",
    "describe", "detect", "identify", "translate",
    "summary", "stop", "pause", "resume",
    "cancel", "repeat", "louder", "softer",
    "mute", "emergency", "call", "help",
    "yes", "no", "setting", "home",
    "sleep", "status", "_silence_", "_unknown_",
};

static void recv_cb(const struct device *mhuv2_ipmdev, void *user_data,
                    uint32_t id, volatile void *data)
{
    ARG_UNUSED(mhuv2_ipmdev);
    ARG_UNUSED(user_data);
    /* DEBUG(MHU 무결성 검증용, 운영 배포 시 제거 가능): HP 수신 누적 카운터. */
    uint32_t wire = *((uint32_t *)data);
#if defined(CONFIG_KWS_MEASURE_MODE)
    uint32_t sc = (wire >> 24) & 0xFFu;
    uint32_t win = (wire >> 8) & 0xFFFFu;
    uint32_t lo_m = wire & 0xFFu;
    uint32_t kw_id_m = (lo_m > 0u) ? (lo_m - 1u) : 0xFFFFFFFFu;
    const char *name_m = (kw_id_m < 32u) ? kw_labels[kw_id_m] : "none";
    if (kw_id_m != 30u) {  /* _silence_ 제외, 실제 발화만 로그 (측정 분석용) */
        printk("[meas] win=%u kw_id=%u (%s) score=%u\n", win, kw_id_m, name_m, sc);
    }
    return;
#else
    static uint32_t hp_rcvd = 0u;
    uint32_t lo = wire & 0xFFu;                 /* kw_id+1 */
    uint32_t he_sent = (wire >> 8) & 0xFFFFFFu; /* HE 송신 카운터 */
    uint32_t kw_id = (lo > 0u) ? (lo - 1u) : 0xFFFFFFFFu;
    const char *name = (kw_id < 32u) ? kw_labels[kw_id] : "?";
    hp_rcvd++;

    printk("RTSS-HP: kw_id=%u (%s) | HE_sent=%u HP_rcvd=%u\n", kw_id, name, he_sent, hp_rcvd);
    if (kw_id == 0u) {
        printk("RTSS-HP: >>> ORINU wake word! (camera trigger placeholder)\n");
    }
#endif
}

int main(void)
{
    uint32_t recv_data;
    printk("RTSS-HP KWS receiver on %s\n", CONFIG_BOARD);
    mhu0_r = DEVICE_DT_GET(DT_ALIAS(rtsshemhu0r));
    if (!device_is_ready(mhu0_r)) {
        printk("MHU0 RX not ready\n");
        return -1;
    }
    ipm_register_callback(mhu0_r, recv_cb, &recv_data);
    ipm_set_enabled(mhu0_r, true);
    printk("RTSS-HP: waiting for KWS doorbell...\n");
    while (1) {
        k_sleep(K_MSEC(1000));
    }
    return 0;
}
