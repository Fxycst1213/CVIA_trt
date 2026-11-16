#include <linux/module.h>
#define INCLUDE_VERMAGIC
#include <linux/build-salt.h>
#include <linux/elfnote-lto.h>
#include <linux/vermagic.h>
#include <linux/compiler.h>

BUILD_SALT;
BUILD_LTO_INFO;

MODULE_INFO(vermagic, VERMAGIC_STRING);
MODULE_INFO(name, KBUILD_MODNAME);

__visible struct module __this_module
__section(".gnu.linkonce.this_module") = {
	.name = KBUILD_MODNAME,
	.init = init_module,
#ifdef CONFIG_MODULE_UNLOAD
	.exit = cleanup_module,
#endif
	.arch = MODULE_ARCH_INIT,
};

#ifdef CONFIG_RETPOLINE
MODULE_INFO(retpoline, "Y");
#endif

static const struct modversion_info ____versions[]
__used __section("__versions") = {
	{ 0x25f8bfc1, "module_layout" },
	{ 0x606d49f3, "usb_serial_generic_tiocmiwait" },
	{ 0x9783df10, "usb_serial_deregister_drivers" },
	{ 0xfd26958d, "usb_serial_register_drivers" },
	{ 0xca192d01, "usb_serial_generic_open" },
	{ 0x30e74134, "tty_termios_copy_hw" },
	{ 0x409873e3, "tty_termios_baud_rate" },
	{ 0x6c257ac0, "tty_termios_hw_change" },
	{ 0x3b5c5334, "tty_put_char" },
	{ 0x54496b4, "schedule_timeout_interruptible" },
	{ 0x15ba50a6, "jiffies" },
	{ 0x85fd2dbe, "_dev_info" },
	{ 0xeb440f45, "usb_serial_generic_resume" },
	{ 0xd58d4b67, "kmem_cache_alloc_trace" },
	{ 0xc1c7e6c0, "kmalloc_caches" },
	{ 0x27a2116c, "usb_control_msg" },
	{ 0x89c9a638, "_dev_err" },
	{ 0x549ff25b, "tty_kref_put" },
	{ 0x6c4cd59e, "usb_serial_handle_dcd_change" },
	{ 0xf093ec8a, "tty_port_tty_get" },
	{ 0x3eeb2322, "__wake_up" },
	{ 0x46834faf, "usb_submit_urb" },
	{ 0xd4bacb8, "usb_kill_urb" },
	{ 0xebebcd5, "usb_serial_generic_close" },
	{ 0x37a0cba, "kfree" },
	{ 0xd35cce70, "_raw_spin_unlock_irqrestore" },
	{ 0x34db050b, "_raw_spin_lock_irqsave" },
	{ 0x1fdc7df2, "_mcount" },
};

MODULE_INFO(depends, "usbserial");

MODULE_ALIAS("usb:v1A86p5523d*dc*dsc*dp*ic*isc*ip*in*");
MODULE_ALIAS("usb:v1A86p7522d*dc*dsc*dp*ic*isc*ip*in*");
MODULE_ALIAS("usb:v1A86p7523d*dc*dsc*dp*ic*isc*ip*in*");
MODULE_ALIAS("usb:v2184p0057d*dc*dsc*dp*ic*isc*ip*in*");
MODULE_ALIAS("usb:v4348p5523d*dc*dsc*dp*ic*isc*ip*in*");
MODULE_ALIAS("usb:v9986p7523d*dc*dsc*dp*ic*isc*ip*in*");
