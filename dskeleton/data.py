import pkgutil
import json

SKELETONS = dict(
    UDH_UPPER = json.loads(
        pkgutil.get_data(__name__, "data/udh_upper.json")),
    BODY_25 = json.loads(
        pkgutil.get_data(__name__, "data/body_25.json"))
)
