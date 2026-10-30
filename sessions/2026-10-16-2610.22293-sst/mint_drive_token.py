"""One-time: mint drive_token.json for Colab checkpoint sync.

Run ONCE on your own machine (not the VM, not Colab):

  1. Create an OAuth client (type "Desktop app") at
     https://console.cloud.google.com/apis/credentials
     and enable the Google Drive API for the project.
  2. python3 mint_drive_token.py --client-id <ID> --client-secret <SECRET>
     -> a browser window opens; approve Drive file access for yourself.
  3. It writes drive_token.json next to this script.

Then, when launching a training driver, upload drive_token.json to the Colab
VM as /content/drive_token.json alongside the driver. The driver refreshes
the access token itself; the refresh token lives only on the VM for that run.

NEVER commit drive_token.json to any repo. Add it to .gitignore everywhere.
Scope is drive.file: the driver can only touch files IT created, never your
whole Drive.
"""

import argparse
import http.server
import json
import threading
import urllib.parse
import urllib.request
import webbrowser

SCOPES = "https://www.googleapis.com/auth/drive.file"
REDIRECT = "http://127.0.0.1:8765/"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--client-id", required=True)
    ap.add_argument("--client-secret", required=True)
    ap.add_argument("--out", default="drive_token.json")
    a = ap.parse_args()

    code_holder = {}

    class H(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            q = urllib.parse.parse_qs(urllib.parse.urlparse(self.path).query)
            code_holder["code"] = q.get("code", [None])[0]
            self.send_response(200)
            self.send_header("Content-Type", "text/plain")
            self.end_headers()
            self.wfile.write(b"Approved. You can close this tab.")
            threading.Thread(target=self.server.shutdown,
                             daemon=True).start()

        def log_message(self, *x):
            pass

    auth = ("https://accounts.google.com/o/oauth2/v2/auth?" +
            urllib.parse.urlencode({
                "client_id": a.client_id,
                "redirect_uri": REDIRECT,
                "response_type": "code",
                "scope": SCOPES,
                "access_type": "offline",
                "prompt": "consent",
            }))
    print("Opening browser for approval...")
    webbrowser.open(auth)
    srv = http.server.HTTPServer(("127.0.0.1", 8765), H)
    srv.serve_forever()
    code = code_holder.get("code")
    if not code:
        raise SystemExit("no auth code received")

    data = urllib.parse.urlencode({
        "code": code,
        "client_id": a.client_id,
        "client_secret": a.client_secret,
        "redirect_uri": REDIRECT,
        "grant_type": "authorization_code",
    }).encode()
    resp = json.load(urllib.request.urlopen(
        urllib.request.Request("https://oauth2.googleapis.com/token",
                               data=data), timeout=60))
    out = {"client_id": a.client_id,
           "client_secret": a.client_secret,
           "refresh_token": resp["refresh_token"]}
    json.dump(out, open(a.out, "w"), indent=1)
    print("wrote %s (refresh token; keep it private)" % a.out)


if __name__ == "__main__":
    main()
