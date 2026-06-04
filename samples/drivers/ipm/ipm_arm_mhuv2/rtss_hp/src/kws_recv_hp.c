/* HP-side KWS receiver: receive kw_id from HE via MHU0, log it with label.
 * (Placeholder for future camera trigger on 'orinu')
 */
#include <zephyr/kernel.h>
#include <zephyr/sys/printk.h>
#include <zephyr/drivers/ipm.h>

const struct device *mhu0_r;

/* Keyword labels, index = kw_id. Must match HE Labels.cc order. */
static const char *kw_labels[66] = {
    "orinu", "home", "reader", "embosser",
    "inkjet", "eyecam", "setting", "start",
    "stop", "pause", "resume", "cancel",
    "louder", "softer", "quiet", "mute",
    "unmute", "status", "help", "play",
    "next", "previous", "repeat", "emergency",
    "test", "eject", "load", "front",
    "back", "both", "yes", "no",
    "shuffle", "timer", "alarm", "ask",
    "connect", "sleep", "wake", "shutdown",
    "read", "capture", "bookmark", "translate",
    "summary", "speed_up", "slow_down", "color",
    "draft", "quality", "duplex", "photo",
    "describe", "zoom_in", "zoom_out", "record",
    "detect", "identify", "call", "wifi",
    "language", "volume", "update", "reset",
    "_silence_", "_unknown_",
};

static void recv_cb(const struct device *mhuv2_ipmdev, void *user_data,
                    uint32_t id, volatile void *data)
{
    ARG_UNUSED(mhuv2_ipmdev);
    ARG_UNUSED(user_data);
    uint32_t wire = *((uint32_t *)data);
    uint32_t kw_id = (wire > 0u) ? (wire - 1u) : 0xFFFFFFFFu; /* undo +1 encoding */
    const char *name = (kw_id < 66u) ? kw_labels[kw_id] : "?";

    printk("RTSS-HP: KWS rcvd kw_id=%u (%s)\n", kw_id, name);
    if (kw_id == 0u) {
        printk("RTSS-HP: >>> ORINU wake word! (camera trigger placeholder)\n");
    }
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
