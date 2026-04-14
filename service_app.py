from utils.bootstrap import ensure_user_site_packages

ensure_user_site_packages()

from service.server import create_app


app = create_app()
