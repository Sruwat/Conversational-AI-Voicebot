import os
import site
import sys


def ensure_user_site_packages():
    try:
        user_site = site.getusersitepackages()
    except Exception:
        return

    if user_site and os.path.isdir(user_site) and user_site not in sys.path:
        sys.path.append(user_site)
