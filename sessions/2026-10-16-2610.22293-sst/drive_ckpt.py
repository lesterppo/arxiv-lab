"""Drive-persistent checkpoints for Colab training drivers (stdlib only).

Problem: free-tier Colab reclaims the VM and wipes /content, so phase markers
and adapter weights saved under /content/<run>/ die with the session. This
module mirrors the run dir to the user's Google Drive so a relaunched session
restores completed phases instead of starting from zero.

Two backends, tried in order:
  1. Drive FUSE mount: /content/drive/MyDrive/arxiv-lab-ckpts/<session>/
     (env DRIVE_CKPT_DIR overrides the root).
  2. Drive REST API over stdlib urllib, using an OAuth2 refresh token in
     drive_token.json (env DRIVE_TOKEN_JSON, default /content/drive_token.json).
     The token file is uploaded to the VM at launch time and is NEVER committed
     to any repo (see mint_drive_token.py, run once on the user's own machine).

API:
    restore(local_dir, session) -> bool   # remote -> local, fresh VM only
    sync(local_dir, session)    -> bool   # local -> remote, skips unchanged

All failures are non-fatal: they log to stderr and return False, and the
driver continues with /content-only checkpoints (current behaviour).
Upload together with the driver: the launch script must copy this file to
/content/ next to the driver.
"""

import io
import json
import os
import sys
import time
import urllib.parse
import urllib.request

MANIFEST = ".ckpt_manifest.json"
TOKEN_ENV = "DRIVE_TOKEN_JSON"
TOKEN_DEFAULT = "/content/drive_token.json"
MOUNT_ROOT = "/content/drive/MyDrive/arxiv-lab-ckpts"


def _log(msg):
    print("[drive_ckpt] %s" % msg, file=sys.stderr, flush=True)


def _manifest(local_dir):
    p = os.path.join(local_dir, MANIFEST)
    try:
        return json.load(open(p))
    except Exception:
        return {}


def _save_manifest(local_dir, m):
    try:
        json.dump(m, open(os.path.join(local_dir, MANIFEST), "w"))
    except Exception as e:
        _log("manifest write failed: %s" % e)


def _scan(local_dir):
    """relpath -> (size, mtime_ns) for all files except the manifest."""
    out = {}
    for root, _dirs, files in os.walk(local_dir):
        for f in files:
            if f == MANIFEST:
                continue
            full = os.path.join(root, f)
            rel = os.path.relpath(full, local_dir)
            try:
                st = os.stat(full)
                out[rel] = (st.st_size, st.st_mtime_ns)
            except OSError:
                pass
    return out


# ---------------------------------------------------------------- mount backend
def _mount_dir(session):
    # Only trust the mount backend when /content/drive is a REAL mount.
    # Without the FUSE mount, makedirs would fabricate a local directory
    # tree and syncs would silently die with the VM.
    if "DRIVE_CKPT_DIR" not in os.environ:
        if not os.path.ismount("/content/drive"):
            return None
        if not os.path.isdir("/content/drive/MyDrive"):
            return None
    root = os.environ.get("DRIVE_CKPT_DIR", MOUNT_ROOT)
    d = os.path.join(root, session)
    try:
        os.makedirs(d, exist_ok=True)
        probe = os.path.join(d, ".write_test")
        with open(probe, "w") as f:
            f.write("ok")
        os.remove(probe)
        return d
    except Exception:
        return None


def _mount_sync(local_dir, remote_dir):
    # Size-keyed comparison: mtimes are meaningless across VM relaunches
    # (restored files get fresh mtimes), so only size decides "unchanged".
    remote_manifest = _manifest(remote_dir)
    local = _scan(local_dir)
    changed = [r for r, sm in local.items()
               if remote_manifest.get(r, [None])[0] != sm[0]]
    for rel in changed:
        src = os.path.join(local_dir, rel)
        dst = os.path.join(remote_dir, rel)
        os.makedirs(os.path.dirname(dst) or remote_dir, exist_ok=True)
        with open(src, "rb") as fsrc, open(dst, "wb") as fdst:
            while True:
                chunk = fsrc.read(8 << 20)
                if not chunk:
                    break
                fdst.write(chunk)
        _log("uploaded %s (%d bytes)" % (rel, local[rel][0]))
    if changed:
        _save_manifest(remote_dir, {r: [sm[0], 0] for r, sm in local.items()})
    return True


def _mount_restore(local_dir, remote_dir):
    local = _scan(local_dir)
    if local:
        return False  # not a fresh VM; local state wins
    remote = _scan(remote_dir)
    if not remote:
        return False
    for rel in remote:
        src = os.path.join(remote_dir, rel)
        dst = os.path.join(local_dir, rel)
        os.makedirs(os.path.dirname(dst) or local_dir, exist_ok=True)
        with open(src, "rb") as fsrc, open(dst, "wb") as fdst:
            while True:
                chunk = fsrc.read(8 << 20)
                if not chunk:
                    break
                fdst.write(chunk)
    _save_manifest(local_dir, _manifest(remote_dir))
    _log("restored %d files from Drive mount" % len(remote))
    return True


# ---------------------------------------------------------------- REST API backend
class _DriveAPI(object):
    def __init__(self, token_path):
        info = json.load(open(token_path))
        self.client_id = info["client_id"]
        self.client_secret = info["client_secret"]
        self.refresh_token = info["refresh_token"]
        self.session = None
        self._access = None
        self._access_exp = 0.0

    def _token(self):
        if time.time() < self._access_exp - 60 and self._access:
            return self._access
        data = urllib.parse.urlencode({
            "client_id": self.client_id,
            "client_secret": self.client_secret,
            "refresh_token": self.refresh_token,
            "grant_type": "refresh_token",
        }).encode()
        req = urllib.request.Request("https://oauth2.googleapis.com/token",
                                     data=data)
        resp = json.load(urllib.request.urlopen(req, timeout=60))
        self._access = resp["access_token"]
        self._access_exp = time.time() + int(resp.get("expires_in", 3600))
        return self._access

    def _req(self, url, method="GET", data=None, headers=None, ctype=None):
        h = {"Authorization": "Bearer " + self._token()}
        if ctype:
            h["Content-Type"] = ctype
        if headers:
            h.update(headers)
        req = urllib.request.Request(url, data=data, headers=h, method=method)
        with urllib.request.urlopen(req, timeout=120) as r:
            body = r.read()
        return json.loads(body) if body else {}

    def _folder(self, name, parent=None):
        q = ("name='%s' and mimeType='application/vnd.google-apps.folder' "
             "and trashed=false" % name.replace("'", "\\'"))
        if parent:
            q += " and '%s' in parents" % parent
        res = self._req("https://www.googleapis.com/drive/v3/files?" +
                        urllib.parse.urlencode(
                            {"q": q, "fields": "files(id)", "pageSize": 5}))
        files = res.get("files", [])
        if files:
            return files[0]["id"]
        body = {"name": name, "mimeType": "application/vnd.google-apps.folder"}
        if parent:
            body["parents"] = [parent]
        res = self._req("https://www.googleapis.com/drive/v3/files",
                        method="POST",
                        data=json.dumps(body).encode(),
                        ctype="application/json")
        return res["id"]

    def root(self, session):
        top = self._folder("arxiv-lab-ckpts")
        return self._folder(session, parent=top)

    def _list(self, folder_id):
        out, token = [], None
        while True:
            params = {"q": "'%s' in parents and trashed=false" % folder_id,
                      "fields": "files(id,name,size,modifiedTime),"
                                "nextPageToken",
                      "pageSize": 200}
            if token:
                params["pageToken"] = token
            res = self._req("https://www.googleapis.com/drive/v3/files?" +
                            urllib.parse.urlencode(params))
            out.extend(res.get("files", []))
            token = res.get("nextPageToken")
            if not token:
                break
        return {f["name"]: f for f in out}

    def _upload(self, folder_id, name, path, existing_id=None):
        meta = {"name": name}
        if not existing_id:
            meta["parents"] = [folder_id]
        boundary = "ckpt-%d" % int(time.time() * 1000)
        buf = io.BytesIO()
        buf.write(("--%s\r\n" % boundary).encode())
        buf.write(b"Content-Type: application/json; charset=UTF-8\r\n\r\n")
        buf.write(json.dumps(meta).encode())
        buf.write(("\r\n--%s\r\n" % boundary).encode())
        buf.write(b"Content-Type: application/octet-stream\r\n\r\n")
        with open(path, "rb") as f:
            while True:
                chunk = f.read(8 << 20)
                if not chunk:
                    break
                buf.write(chunk)
        buf.write(("\r\n--%s--\r\n" % boundary).encode())
        body = buf.getvalue()
        if existing_id:
            url = ("https://www.googleapis.com/upload/drive/v3/files/%s"
                   "?uploadType=multipart" % existing_id)
            self._req(url, method="PATCH", data=body,
                      ctype="multipart/related; boundary=" + boundary)
        else:
            url = ("https://www.googleapis.com/upload/drive/v3/files"
                   "?uploadType=multipart")
            self._req(url, method="POST", data=body,
                      ctype="multipart/related; boundary=" + boundary)

    def _download(self, file_id, dest):
        url = ("https://www.googleapis.com/drive/v3/files/%s?alt=media"
               % file_id)
        h = {"Authorization": "Bearer " + self._token()}
        req = urllib.request.Request(url, headers=h)
        os.makedirs(os.path.dirname(dest) or ".", exist_ok=True)
        with urllib.request.urlopen(req, timeout=300) as r, \
                open(dest, "wb") as f:
            while True:
                chunk = r.read(8 << 20)
                if not chunk:
                    break
                f.write(chunk)

    def sync(self, local_dir):
        folder = self.root(self.session)
        remote = self._list(folder)
        manifest = {}
        try:
            m_id = remote.get(MANIFEST, {}).get("id")
            if m_id:
                tmp = os.path.join(local_dir, MANIFEST + ".remote")
                self._download(m_id, tmp)
                manifest = json.load(open(tmp))
                os.remove(tmp)
        except Exception:
            manifest = {}
        local = _scan(local_dir)
        for rel, (size, _mt) in local.items():
            entry = manifest.get(rel)
            if entry and entry[0] == size:
                continue  # unchanged (size-keyed; mtime differs across VMs)
            name = rel.replace(os.sep, "__")
            self._upload(folder, name, os.path.join(local_dir, rel),
                         existing_id=remote.get(name, {}).get("id"))
            _log("uploaded %s (%d bytes)" % (rel, size))
            manifest[rel] = [size, 0]
        mpath = os.path.join(local_dir, MANIFEST)
        json.dump(manifest, open(mpath, "w"))
        self._upload(folder, MANIFEST, mpath,
                     existing_id=remote.get(MANIFEST, {}).get("id"))
        return True

    def restore(self, local_dir):
        if _scan(local_dir):
            return False  # not a fresh VM; local state wins
        folder = self.root(self.session)
        remote = self._list(folder)
        if not remote:
            return False
        manifest = {}
        try:
            m_id = remote.get(MANIFEST, {}).get("id")
            if m_id:
                tmp = os.path.join(local_dir, MANIFEST + ".remote")
                self._download(m_id, tmp)
                manifest = json.load(open(tmp))
                os.remove(tmp)
        except Exception:
            pass
        names = manifest.keys() if manifest else \
            [n for n in remote if n != MANIFEST]
        n = 0
        for rel in names:
            name = rel.replace(os.sep, "__")
            fid = remote.get(name, {}).get("id")
            if not fid:
                continue
            dest = os.path.join(local_dir, rel)
            os.makedirs(os.path.dirname(dest) or local_dir, exist_ok=True)
            self._download(fid, dest)
            n += 1
        _log("restored %d files from Drive API" % n)
        return n > 0


def _api_client(session):
    path = os.environ.get(TOKEN_ENV, TOKEN_DEFAULT)
    if not os.path.exists(path):
        return None
    try:
        api = _DriveAPI(path)
        api.session = session
        api._token()  # fail fast on bad/expired refresh token
        return api
    except Exception as e:
        _log("Drive API unavailable: %s" % e)
        return None


# ---------------------------------------------------------------- public API
def restore(local_dir, session):
    """Pull checkpoints from Drive on a fresh VM. Returns True if restored."""
    try:
        mnt = _mount_dir(session)
        if mnt:
            return _mount_restore(local_dir, mnt)
        api = _api_client(session)
        if api:
            return api.restore(local_dir)
        _log("no Drive backend available; starting fresh")
        return False
    except Exception as e:
        _log("restore failed: %s" % e)
        return False


def sync(local_dir, session):
    """Push local checkpoints to Drive. Returns True on success."""
    try:
        mnt = _mount_dir(session)
        if mnt:
            return _mount_sync(local_dir, mnt)
        api = _api_client(session)
        if api:
            return api.sync(local_dir)
        _log("no Drive backend available; checkpoints stay local")
        return False
    except Exception as e:
        _log("sync failed: %s" % e)
        return False
