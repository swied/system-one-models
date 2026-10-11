• The official guidance points to two common causes: bubblewrap is missing, or Ubuntu’s AppArmor policy is blocking unprivileged user namespaces. I’m checking which case applies in this Linux environment so I can tailor the commands.

───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────

• That warning means Codex cannot start its Linux sandbox. Run the fix in a regular terminal outside Codex, then restart Codex.

  For Ubuntu/Debian:

  sudo apt update
  sudo apt install bubblewrap

  For Fedora:

  sudo dnf install bubblewrap

  If you’re on Ubuntu 24.04 and the warning remains, load the recommended AppArmor profile:

  sudo apt install apparmor-profiles apparmor-utils

  sudo install -m 0644 \
    /usr/share/apparmor/extra-profiles/bwrap-userns-restrict \
    /etc/apparmor.d/bwrap-userns-restrict

  sudo apparmor_parser -r /etc/apparmor.d/bwrap-userns-restrict

  Then restart Codex. Ubuntu 25.04 and later should normally need only the bubblewrap installation.

  As a last resort, you can temporarily disable Ubuntu’s global AppArmor restriction:

  sudo sysctl -w kernel.apparmor_restrict_unprivileged_userns=0

  That weakens a system-wide security restriction, so prefer the targeted AppArmor profile. The official OpenAI documentation recommends this same order of fixes: Codex sandboxing (https://learn.chatgpt.com/docs/sandboxing).