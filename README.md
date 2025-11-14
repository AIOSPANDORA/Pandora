# AIOS Pandora (Cubic USB/Live Boot Ready)

## Setup

1. **Integrate into ISO**  
   Use [Cubic](https://launchpad.net/cubic) to open your base Ubuntu ISO.  
   Copy the `aios-pandora-usb/` folder to `/` of your live ISO environment during Cubic setup.

2. **Set Permissions**  
   Run `chmod +x /aios-pandora-usb/*.sh /aios-pandora-usb/startup/*.py`.

3. **Autostart**  
   Copy `aios-autostart.desktop` to `/etc/skel/.config/autostart/` during Cubic.

4. **(Optional) Headless/Daemon Mode**  
   Copy the sample systemd unit into `/etc/systemd/system/`, then  
   `systemctl enable pandora-aios` in Cubic.

5. **Burn ISO to USB & Boot!**  
   Use Etcher or Rufus to create your bootable USB from the finished ISO.

## On Boot

- Antivirus and firewalls start before AIOS.
- Pandora always runs safe by default; full orchestrator only starts if health checks pass.