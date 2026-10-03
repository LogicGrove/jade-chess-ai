from pathlib import Path
import json
import sys
REPO_ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(REPO_ROOT/'src'))
import Jade_Programa as j
from playwright.sync_api import sync_playwright

with sync_playwright() as p:
    browser=p.chromium.launch(headless=True,args=["--no-sandbox"])
    errors=[]
    page=browser.new_page(viewport={"width":360,"height":850},device_scale_factor=1)
    page.on("pageerror",lambda exc:errors.append(str(exc)))
    page.set_content(Path("qa_report.html").read_text(),wait_until="load")
    assert page.locator("#board svg").count()==1
    first=page.locator("#eval-caption").inner_text()
    page.locator("#line").click()
    assert "alternativa" in page.locator("#eval-caption").inner_text()
    page.locator("#back").click()
    assert "Antes" in page.locator("#eval-caption").inner_text()
    for _ in range(3): page.locator("#next").click()
    assert "negras" in page.locator("#eval-caption").inner_text()
    assert page.locator("#evalwhite").evaluate("e=>e.style.height")=="0%"
    assert page.locator("#eval-label").inner_text()=="-#0"
    assert page.evaluate("document.documentElement.scrollWidth<=window.innerWidth")
    page.screenshot(path="qa_pgn_mobile.png",full_page=True)
    page.set_viewport_size({"width":1200,"height":900})
    assert page.evaluate("document.documentElement.scrollWidth<=window.innerWidth")
    page.screenshot(path="qa_pgn_desktop.png",full_page=True)
    document=Path("qa_report.html").read_text()
    data=json.JSONDecoder().raw_decode(document.split("const d=",1)[1])[0]
    row=dict(data["rows"][0],played_score=dict(cp=250,mate=None,value=250),evaluation="+2.50")
    page.set_content(j.jade_report_html([row],data["metadata"]))
    assert float(page.locator("#evalwhite").evaluate("e=>e.style.height").rstrip("%"))>50
    assert "+250 cp" in page.locator("#eval-caption").inner_text()
    row.update(played_score=dict(cp=None,mate=0,value=100000),evaluation="#0")
    page.set_content(j.jade_report_html([row],data["metadata"]))
    assert page.locator("#evalwhite").evaluate("e=>e.style.height")=="100%"
    assert not errors,errors
    browser.close()
print("PASS: SVG, barra, navegación, mate negro y ancho móvil/escritorio sin errores JS.")
