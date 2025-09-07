from kedro.config import OmegaConfigLoader
from kedro.framework.project import settings
import os
# Instantiate an `OmegaConfigLoader` instance with the location of your project configuration.
conf_path = str(f"{os.getcwd()}/{settings.CONF_SOURCE}")
conf_loader = OmegaConfigLoader(conf_source=conf_path)

conf_catalog = conf_loader["catalog"]
