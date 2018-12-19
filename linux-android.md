sudo apt-get install android-tools-adb

lsusb

adb devices

sudo vim /etc/udev/rules.d/51-android.rules

SUBSYSTEM=="usb", ENV{DEVTYPE}=="usb_device", MODE="0666"

sudo /etc/init.d/udev restart
