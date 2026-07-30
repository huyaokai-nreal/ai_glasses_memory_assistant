#!/bin/sh
set -eu

serial="${1:?usage: export_s22_model_pack.sh ANDROID_SERIAL}"
package="com.aiglasses.memoryassistant.demo"
pack="x4000-sherpa-1.13.4-v2-4bba9209449c"
destination="$(cd "$(dirname "$0")/.." && pwd)/.local-models/x4000-sherpa-1.13.4-v2"

mkdir -p "$destination"
adb -s "$serial" exec-out run-as "$package" /system/bin/sh -c "cd files/models/packs/$pack && tar -cf - ." | tar -xf - -C "$destination"
python3 "$(dirname "$0")/verify_model_pack.py" --pack-dir "$destination"
