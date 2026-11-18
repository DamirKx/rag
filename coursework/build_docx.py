"""Сборка черновика курсовой из Markdown в .docx по ГОСТ 7.32 и подсчёт страниц в Word.

    pip install pypandoc_binary python-docx
    python coursework/build_docx.py coursework/v20 coursework/v25

В каждой папке ищутся 01_*.md … 05_*.md; рядом создаётся kursovaya.docx. Титульный лист и содержание
добавляются заглушками (по одной странице), чтобы число страниц было как в итоговом документе.
Если установлен Word (Windows), рядом сохраняется PDF и печатается число страниц.
"""
from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

import pypandoc
from docx import Document
from docx.enum.section import WD_SECTION
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_BREAK, WD_LINE_SPACING
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Pt, RGBColor

FONT = "Times New Roman"
FIGURE_HEIGHT_CM = 8  # место под рисунок 1 (в черновике — текстовая заглушка)


def read_markdown(folder: Path) -> str:
    parts = [p.read_text(encoding="utf-8") for p in sorted(folder.glob("0[1-5]_*.md"))]
    if not parts:
        raise SystemExit(f"В {folder} нет файлов 01_*.md … 05_*.md")
    text = "\n\n".join(parts)
    # «(N)» — номер формулы справа: оставляем его внутри формулы через пробел
    text = re.sub(r"\\qquad\s*\((\d+)\)", r"\\quad\\quad\\quad(\1)", text)
    return text


def set_font(run, size=14, bold=None):
    run.font.name = FONT
    run.font.size = Pt(size)
    run._element.get_or_add_rPr().get_or_add_rFonts().set(qn("w:eastAsia"), FONT)
    if bold is not None:
        run.bold = bold


def format_paragraph(p, *, size=14, indent=True, align=WD_ALIGN_PARAGRAPH.JUSTIFY, spacing=1.5):
    pf = p.paragraph_format
    pf.line_spacing_rule = WD_LINE_SPACING.ONE_POINT_FIVE if spacing == 1.5 else WD_LINE_SPACING.SINGLE
    pf.space_before = Pt(0)
    pf.space_after = Pt(0)
    pf.first_line_indent = Cm(1.25) if indent else Cm(0)
    pf.left_indent = Cm(0)
    p.alignment = align
    for run in p.runs:
        set_font(run, size)


def page_break_before(p):
    p.paragraph_format.page_break_before = True


def placeholder_page(doc: Document, title: str, before):
    p = before.insert_paragraph_before(title)
    format_paragraph(p, indent=False, align=WD_ALIGN_PARAGRAPH.CENTER)
    for run in p.runs:
        run.bold = True
    br = before.insert_paragraph_before()
    br.add_run().add_break(WD_BREAK.PAGE)


def style_document(path: Path):
    doc = Document(str(path))
    sec = doc.sections[0]
    sec.page_height, sec.page_width = Cm(29.7), Cm(21.0)
    sec.left_margin, sec.right_margin = Cm(3.0), Cm(1.5)
    sec.top_margin, sec.bottom_margin = Cm(2.0), Cm(2.0)

    body = doc.paragraphs
    first = body[0]
    for p in body:
        style = p.style.name
        text = p.text.strip()
        if style.startswith("Heading"):
            level = int(style.split()[-1]) if style[-1].isdigit() else 1
            upper = level == 1 and not re.match(r"^\d", text)
            format_paragraph(p, indent=not upper, align=WD_ALIGN_PARAGRAPH.CENTER if upper else WD_ALIGN_PARAGRAPH.LEFT)
            p.paragraph_format.space_before = Pt(12 if level > 1 else 0)
            p.paragraph_format.space_after = Pt(12)
            p.paragraph_format.keep_with_next = True
            for run in p.runs:
                set_font(run, 14, bold=True)
                run.font.color.rgb = RGBColor(0, 0, 0)
                run.italic = False
            if level == 1:
                page_break_before(p)
        elif text.startswith("Таблица ") and "–" in text[:14]:
            format_paragraph(p, indent=False, align=WD_ALIGN_PARAGRAPH.LEFT)
            p.paragraph_format.keep_with_next = True
            p.paragraph_format.space_before = Pt(6)
        elif text.startswith("[Рисунок"):
            caption = text.strip("[]")
            p.text = caption.split(":")[0]
            format_paragraph(p, indent=False, align=WD_ALIGN_PARAGRAPH.CENTER)
            p.paragraph_format.space_before = Cm(FIGURE_HEIGHT_CM)
        elif style in ("Compact", "List Paragraph") or p._p.pPr is not None and p._p.pPr.numPr is not None:
            format_paragraph(p, indent=False)
            p.paragraph_format.left_indent = Cm(1.25)
            p.paragraph_format.first_line_indent = Cm(-0.6)
        else:
            format_paragraph(p)
        # формулы (oMath) — по центру
        if p._p.find(qn("m:oMathPara")) is not None:
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            p.paragraph_format.first_line_indent = Cm(0)

    for table in doc.tables:
        table.alignment = WD_TABLE_ALIGNMENT.CENTER
        tbl_pr = table._tbl.tblPr
        borders = OxmlElement("w:tblBorders")
        for edge in ("top", "left", "bottom", "right", "insideH", "insideV"):
            el = OxmlElement(f"w:{edge}")
            el.set(qn("w:val"), "single")
            el.set(qn("w:sz"), "4")
            el.set(qn("w:color"), "000000")
            borders.append(el)
        tbl_pr.append(borders)
        for row in table.rows:
            for cell in row.cells:
                for p in cell.paragraphs:
                    format_paragraph(p, size=12, indent=False, align=WD_ALIGN_PARAGRAPH.LEFT, spacing=1)

    # титульный лист и содержание — заглушки по странице
    placeholder_page(doc, "ТИТУЛЬНЫЙ ЛИСТ (по шаблону вуза)", first)
    placeholder_page(doc, "СОДЕРЖАНИЕ", first)
    first.paragraph_format.page_break_before = False
    doc.save(str(path))


def export_pdf(path: Path) -> int | None:
    """Сохраняет PDF рядом с .docx через Word и возвращает число страниц (None — Word недоступен)."""
    pdf = path.with_suffix(".pdf")
    ps = (
        "$w = New-Object -ComObject Word.Application; $w.Visible = $false; "
        f"$d = $w.Documents.Open('{path.resolve()}', $false, $true); "
        f"$d.ExportAsFixedFormat('{pdf.resolve()}', 17); "
        "$n = $d.Content.Information(4); $d.Close($false); $w.Quit(); $n"
    )
    try:
        out = subprocess.run(["powershell", "-NoProfile", "-Command", ps], capture_output=True, text=True, timeout=300)
        return int(out.stdout.strip().splitlines()[-1])
    except Exception:  # Word не установлен или COM недоступен
        return None


def build(folder: Path):
    out = folder / "kursovaya.docx"
    pypandoc.convert_text(read_markdown(folder), "docx", format="markdown", outputfile=str(out),
                          extra_args=["--wrap=none"])
    style_document(out)
    pages = export_pdf(out)
    print(f"{out}: {pages if pages is not None else '?'} стр.")


if __name__ == "__main__":
    for arg in sys.argv[1:] or ["coursework"]:
        build(Path(arg))
