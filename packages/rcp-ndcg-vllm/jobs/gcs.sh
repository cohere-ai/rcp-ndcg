#!/usr/bin/env bash
# The node scripts' GCS transfer dispatch (one home; wave0.sh and bootstrap.sh source it).
#
# The stock engine image ships neither gcloud nor gsutil. The mounted auth script runs FIRST (it is
# executed, never printed, and its output is never echoed; it sets up the credentials every path uses),
# then the transfer path is chosen once and recorded for the report:
#
#   gcs_transfer_detect     -> prints gcloud | gsutil | python
#   gcs_cp SRC DST          -> copy a file or directory, either side gs://
#   gcs_ls URI              -> list a gs:// prefix
#   gcs_rm URI              -> delete one gs:// object
#
# The Python path installs gcsfs into a tools directory OUTSIDE the engine environment (pip --target,
# the same way uv gets installed), from the wheelhouse when one is staged there and the stage is already
# local, else from PyPI (declared, recorded), and runs gcs.py with it. The tools directory and the
# interpreter come from the caller (GCS_TOOLS_DIR, GCS_PYTHON), so no default hides a machine path.

set -euo pipefail

#: The transfer path wave 0 records: gcloud, gsutil, or python (the gcsfs helper).
gcs_transfer_detect() {
  if command -v gcloud >/dev/null; then
    printf 'gcloud\n'
  elif command -v gsutil >/dev/null; then
    printf 'gsutil\n'
  else
    printf 'python\n'
  fi
}

#: Install gcsfs into GCS_TOOLS_DIR (never the engine environment): from a wheelhouse when one is
#: already on local disk (GCS_WHEELHOUSE, the staged one), else from PyPI - the one declared network
#: dependency of the transfer setup, before the stage itself is reachable.
gcs_ensure_tools() {
  local tools="${GCS_TOOLS_DIR:?set GCS_TOOLS_DIR to a tools directory outside the engine environment}"
  if [[ -f "$tools/gcsfs/__init__.py" ]]; then
    return 0
  fi
  mkdir -p "$tools"
  if [[ -n "${GCS_WHEELHOUSE:-}" && -d "$GCS_WHEELHOUSE" ]] && ls "$GCS_WHEELHOUSE"/gcsfs-*.whl >/dev/null 2>&1; then
    "$GCS_PY" -m pip install --quiet --no-index --find-links "$GCS_WHEELHOUSE" --target "$tools" gcsfs
  else
    "$GCS_PY" -m pip install --quiet --target "$tools" gcsfs
  fi
}

#: The python-path invocation of the helper: the tools dir first on sys.path, the engine env untouched.
gcs_run() { # gcs_run GCS_PY_PATH ARGS...
  local helper="${GCS_HELPER_PY:?set GCS_HELPER_PY to the mounted gcs.py}"
  PYTHONPATH="${GCS_TOOLS_DIR}${PYTHONPATH:+:$PYTHONPATH}" "$GCS_PY" "$helper" "$@"
}

gcs_cp() { # gcs_cp SRC DST: copy a file or directory, either side gs://
  case "$(gcs_transfer_detect)" in
    gcloud) gcloud storage cp -r "$1" "$2" ;;
    gsutil) gsutil -m cp -r "$1" "$2" ;;
    *) gcs_ensure_tools; gcs_run cp "$1" "$2" ;;
  esac
}

gcs_ls() { # gcs_ls URI: the entries under a gs:// prefix
  case "$(gcs_transfer_detect)" in
    gcloud) gcloud storage ls "$1" ;;
    gsutil) gsutil ls "$1" ;;
    *) gcs_run ls "$1" ;;
  esac
}

gcs_rm() { # gcs_rm URI: delete one gs:// object
  case "$(gcs_transfer_detect)" in
    gcloud) gcloud storage rm "$1" ;;
    gsutil) gsutil rm "$1" ;;
    *) gcs_run rm "$1" ;;
  esac
}
