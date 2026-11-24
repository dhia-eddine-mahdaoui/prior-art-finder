"""Drive the real web UI in headless Chrome to produce the README screenshots and the demo GIF.

    ./run.sh capture shots      # docs/screenshots/*.png
    ./run.sh capture demo       # docs/demo.gif (+ docs/demo.mp4)

Needs `pip install playwright imageio-ffmpeg pillow` and a Chrome / Chromium. The script starts its own server on a
free port with a throw-away data directory, and stops the server and browser when it is done. Screenshots are page
content only (no browser chrome), so no address bar appears in them.
"""
from __future__ import annotations

import argparse
import os
import shutil
import socket
import subprocess
import sys
import tempfile
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SHOTS = ROOT / "docs" / "screenshots"
DEMO_TEXT = (
    "We propose a thermal management arrangement for a traction battery pack used in electric vehicles. "
    "A phase-change material layer is disposed between adjacent battery cells to absorb heat during fast charging. "
    "Underneath the cells, a cooling plate having serpentine microchannels circulates coolant. "
    "A controller adjusts coolant flow rate using a predictive model of the cell temperature gradient, rather than reacting to absolute temperature. "
    "Thermal runaway is detected when a rate of temperature rise exceeds a threshold, in which case an intumescent barrier and a venting channel direct ejected gas away from adjacent cells. "
    "In cold ambient conditions a reversible heat pump pre-heats the battery pack before charging."
)

CURSOR_JS = """
(() => {
  const c = document.createElement('div');
  c.style.cssText = 'position:fixed;z-index:99999;width:18px;height:18px;border-radius:50%;background:rgba(59,91,219,.55);border:2px solid #fff;box-shadow:0 1px 6px rgba(0,0,0,.35);pointer-events:none;left:-40px;top:-40px;transform:translate(-50%,-50%);transition:transform .08s';
  const add = () => document.body && document.body.appendChild(c);
  document.addEventListener('DOMContentLoaded', add);
  addEventListener('mousemove', e => { c.style.left = e.clientX + 'px'; c.style.top = e.clientY + 'px'; });
  addEventListener('mousedown', () => c.style.transform = 'translate(-50%,-50%) scale(.7)');
  addEventListener('mouseup', () => c.style.transform = 'translate(-50%,-50%) scale(1)');
})();
"""


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("localhost", 0))
        return s.getsockname()[1]


class Server:
    def __init__(self):
        self.port = free_port()
        self.var = tempfile.mkdtemp(prefix="priorart-capture-")
        env = dict(os.environ, PRIORART_VAR=self.var)
        py = ROOT / ".venv" / "bin" / "python"
        self.proc = subprocess.Popen([str(py if py.exists() else sys.executable), "-m", "priorart", "serve", "--port", str(self.port)],
                                     cwd=ROOT, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        self.url = f"http://localhost:{self.port}"
        for _ in range(180):
            try:
                urllib.request.urlopen(self.url + "/api/meta", timeout=2)
                return
            except Exception:
                time.sleep(1)
        self.stop()
        raise RuntimeError("server did not start")

    def stop(self):
        self.proc.terminate()
        try:
            self.proc.wait(15)
        except Exception:
            self.proc.kill()
        shutil.rmtree(self.var, ignore_errors=True)


def launch(pw):
    for kw in ({}, {"executable_path": "/usr/bin/google-chrome"}):
        try:
            return pw.chromium.launch(args=["--no-sandbox"], **kw)
        except Exception:
            continue
    raise RuntimeError("no usable Chrome / Chromium found")


class Driver:
    def __init__(self, page, pace: float = 1.0):
        self.page, self.pace = page, pace

    def pause(self, s: float):
        self.page.wait_for_timeout(int(s * 1000 * self.pace))

    def click(self, sel: str, after: float = 0.5, **kw):
        loc = self.page.locator(sel).first
        loc.scroll_into_view_if_needed()
        box = loc.bounding_box()
        x, y = box["x"] + box["width"] / 2, box["y"] + box["height"] / 2
        self.page.mouse.move(x, y, steps=14 if self.pace > 0 else 1)
        self.pause(0.25)
        self.page.mouse.down()
        self.page.mouse.up()
        self.pause(after)

    def type(self, sel: str, text: str, delay: int = 22):
        self.click(sel, 0.2)
        self.page.keyboard.type(text, delay=int(delay * self.pace))

    def scroll_to(self, sel: str, block="start"):
        self.page.evaluate("([s,b])=>document.querySelector(s).scrollIntoView({behavior:'smooth',block:b})", [sel, block])
        self.pause(0.9)


def new_context(browser, video_dir=None, size=(1360, 860)):
    kw = dict(viewport={"width": size[0], "height": size[1]}, device_scale_factor=1)
    if video_dir:
        kw.update(record_video_dir=str(video_dir), record_video_size={"width": size[0], "height": size[1]})
    ctx = browser.new_context(**kw)
    ctx.add_init_script(CURSOR_JS)
    return ctx


# ------------------------------------------------------------------ screenshots
def shots(url: str):
    from playwright.sync_api import sync_playwright
    SHOTS.mkdir(parents=True, exist_ok=True)
    with sync_playwright() as pw:
        browser = launch(pw)
        try:
            ctx = new_context(browser, size=(1360, 900))
            page = ctx.new_page()
            d = Driver(page, pace=0)
            page.goto(url + "/#/")
            page.wait_for_selector("#new-sample")
            page.select_option("#new-sample", "D1")
            d.click("#new-btn", 0)
            page.wait_for_selector(".fin")
            page.fill("#dpd", "2023-03-10")
            page.dispatch_event("#dpd", "change")
            d.click("#run", 0)
            page.wait_for_selector(".res")
            page.wait_for_timeout(500)
            hide_cursor(page)
            page.screenshot(path=str(SHOTS / "01-search-results.png"))
            # claim-chart style feature mapping of the top hit
            page.evaluate("window.scrollBy(0, document.querySelector('.res').getBoundingClientRect().top - 84)")
            page.wait_for_timeout(300)
            page.screenshot(path=str(SHOTS / "02-feature-mapping.png"))
            # filters + date cut-off
            page.evaluate("window.scrollTo(0,0)")
            page.fill("#fpd", "2022-06-30")
            d.click("#apply", 0)
            page.wait_for_selector(".res")
            page.click("details:has-text('Closest documents filed after')>summary")
            page.wait_for_timeout(300)
            hide_cursor(page)
            page.screenshot(path=str(SHOTS / "03-filters-date-cutoff.png"))
            # shortlist two references, then the shortlist / workflow view
            page.fill("#fpd", "2023-03-10")
            d.click("#apply", 0)
            page.wait_for_selector(".res")
            for _ in range(3):
                page.locator("[data-act=shortlist]").nth(0 if _ == 0 else _).click()
                page.wait_for_timeout(250)
            d.click("#step-shortlist", 0)
            page.wait_for_selector(".dec")
            page.select_option(".dec >> nth=0", "relevant")
            page.wait_for_timeout(400)
            page.fill(".note-in >> nth=0", "Discloses the cooling plate and the heat pump; check the predictive control in the full text.")
            page.dispatch_event(".note-in >> nth=0", "change")
            page.wait_for_timeout(500)
            hide_cursor(page)
            page.screenshot(path=str(SHOTS / "04-disclosure-workflow-shortlist.png"))
            d.click("#tochart", 0)
            page.wait_for_selector(".chart")
            page.wait_for_timeout(400)
            hide_cursor(page)
            page.screenshot(path=str(SHOTS / "05-claim-chart-export.png"))
            page.set_viewport_size({"width": 1360, "height": 1130})
            page.goto(url + "/#/eval")
            page.wait_for_selector(".kpi")
            page.wait_for_timeout(500)
            hide_cursor(page)
            page.screenshot(path=str(SHOTS / "06-evaluation.png"))
        finally:
            browser.close()


def hide_cursor(page):
    page.evaluate("document.querySelectorAll('div').forEach(e=>{if(e.style&&e.style.zIndex==='99999')e.style.display='none'});document.getElementById('toast').style.display='none'")


# ------------------------------------------------------------------ demo recording
def demo(url: str, out_dir: Path):
    from playwright.sync_api import sync_playwright
    import imageio_ffmpeg
    vid = Path(tempfile.mkdtemp(prefix="priorart-video-"))
    with sync_playwright() as pw:
        browser = launch(pw)
        try:
            ctx = new_context(browser, video_dir=vid, size=(1280, 800))
            page = ctx.new_page()
            d = Driver(page, pace=0.7)
            page.goto(url + "/#/")
            page.wait_for_selector("#new-sample")
            d.pause(0.8)
            d.click("#new-btn", 0.6)                                   # new disclosure
            page.wait_for_selector("#dtext")
            d.click("#pname", 0.2)
            page.keyboard.press("Control+A")
            page.keyboard.type("Predictive pack thermal management", delay=25)
            d.type("#dtext", DEMO_TEXT[:120], delay=24)                 # typed, then pasted
            page.keyboard.insert_text(DEMO_TEXT[120:])
            d.pause(0.5)
            d.click("#dpd", 0.2)
            page.fill("#dpd", "2023-03-10")
            page.dispatch_event("#dpd", "change")
            d.click("#dclasses .chip:text-is('H01M')", 0.2)             # blur -> features are extracted
            page.wait_for_selector(".fin")
            d.pause(1.6)
            d.click("#run", 0.4)
            page.wait_for_selector(".res")
            d.pause(1.6)                                                # ranked results with coverage strips
            d.scroll_to(".res .tbl", "center")                          # claim-chart view with highlights
            d.pause(2.0)
            d.scroll_to("#fpd", "start")
            page.evaluate("window.scrollTo({top:0,behavior:'smooth'})")
            d.pause(0.7)
            page.fill("#fpd", "2022-06-30")                             # priority-date cut-off
            d.click("#apply", 0.4)
            page.wait_for_selector(".res")
            d.pause(1.4)
            d.click("[data-act=shortlist] >> nth=0", 0.5)
            d.click("[data-act=shortlist] >> nth=1", 0.5)
            d.click("#step-shortlist", 0.4)
            try:
                page.wait_for_selector(".dec", timeout=8000)
            except Exception:
                page.screenshot(path=os.path.join(tempfile.gettempdir(), "priorart-demo-fail.png"))
                raise
            d.pause(1.4)
            d.click("#tochart", 0.4)
            page.wait_for_selector(".chart")
            d.pause(1.6)
            d.click("#exp-md", 0.3)                                     # export
            d.pause(1.4)
            video = Path(page.video.path())
            ctx.close()
        finally:
            browser.close()
    out_dir.mkdir(parents=True, exist_ok=True)
    ff = imageio_ffmpeg.get_ffmpeg_exe()
    mp4, gif, pal = out_dir / "demo.mp4", out_dir / "demo.gif", vid / "pal.png"
    subprocess.run([ff, "-y", "-loglevel", "error", "-i", str(video), "-vf", "setpts=PTS/1.15,scale=960:-2", "-c:v", "libx264", "-crf", "30",
                    "-pix_fmt", "yuv420p", "-an", str(mp4)], check=True)
    vf = "setpts=PTS/1.15,fps=10,scale=860:-1:flags=lanczos"
    subprocess.run([ff, "-y", "-loglevel", "error", "-i", str(video), "-vf", vf + ",palettegen=max_colors=48:stats_mode=diff", str(pal)], check=True)
    subprocess.run([ff, "-y", "-loglevel", "error", "-i", str(video), "-i", str(pal), "-lavfi",
                    vf + "[x];[x][1:v]paletteuse=dither=none:diff_mode=rectangle", str(gif)], check=True)
    shutil.rmtree(vid, ignore_errors=True)
    print(f"wrote docs/{gif.name} ({gif.stat().st_size / 1e6:.2f} MB) and docs/{mp4.name} ({mp4.stat().st_size / 1e6:.2f} MB)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=("shots", "demo"))
    ap.add_argument("--url", help="use a running server instead of starting one")
    a = ap.parse_args()
    srv = None if a.url else Server()
    try:
        url = a.url or srv.url
        shots(url) if a.what == "shots" else demo(url, ROOT / "docs")
    finally:
        if srv:
            srv.stop()
