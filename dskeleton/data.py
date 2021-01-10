import pkgutil
import json

SKELETONS = dict(
    TEST = json.loads(
        pkgutil.get_data(__name__, "data/test.json")),
    UDH_UPPER = json.loads(
        pkgutil.get_data(__name__, "data/udh_upper.json")),
    UDH_HEAD = json.loads(
        pkgutil.get_data(__name__, "data/udh_head.json")),
    BODY_25 = json.loads(
        pkgutil.get_data(__name__, "data/body_25.json"))
)
