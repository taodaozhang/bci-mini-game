from pathlib import Path
import json
from docx import Document
from docx.shared import Cm, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_CELL_VERTICAL_ALIGNMENT
from docx.oxml import OxmlElement
from docx.oxml.ns import qn

BASE = Path(__file__).resolve().parents[2]
OUT = BASE / 'outputs' / '项目文档'
OUT.mkdir(parents=True, exist_ok=True)
ROOT = r'C:\Users\Lenovo\Desktop\脑机小游戏'
PROJECT = ROOT + r'\eeg_accuracy_upgrade'

def P(s): return ('p', s)
def H(s): return ('h', s)
def B(*s): return ('b', list(s))
def C(s): return ('c', s.strip('\n'))
def T(headers, rows, widths=None): return ('t', headers, rows, widths)
def I(path, caption): return ('i', str(path), caption)

def set_font(style, size, bold=False):
    style.font.name = 'Calibri'
    style.font.size = Pt(size)
    style.font.bold = bold
    style.font.color.rgb = RGBColor(0,0,0)
    rpr = style.element.get_or_add_rPr()
    fonts = rpr.find(qn('w:rFonts'))
    if fonts is None:
        fonts = OxmlElement('w:rFonts'); rpr.insert(0, fonts)
    for k,v in [('ascii','Calibri'),('hAnsi','Calibri'),('eastAsia','Microsoft YaHei')]:
        fonts.set(qn('w:'+k), v)
    for key in ['w:color']:
        el = rpr.find(qn(key))
        if el is not None:
            for attr in ['w:themeColor','w:themeTint','w:themeShade']:
                el.attrib.pop(qn(attr), None)

def shade(cell, fill):
    p = cell._tc.get_or_add_tcPr()
    shd = OxmlElement('w:shd'); shd.set(qn('w:fill'),fill); p.append(shd)

def table(doc, headers, rows, widths):
    t = doc.add_table(rows=1, cols=len(headers))
    t.autofit = False
    widths = widths or [17/len(headers)]*len(headers)
    for c,w in zip(t.columns,widths): c.width=Cm(w)
    borders = OxmlElement('w:tblBorders')
    for side in ['top','left','bottom','right','insideH','insideV']:
        el=OxmlElement('w:'+side); el.set(qn('w:val'),'single'); el.set(qn('w:sz'),'4'); el.set(qn('w:color'),'D9D9D9'); borders.append(el)
    t._tbl.tblPr.append(borders)
    margins=OxmlElement('w:tblCellMar')
    for side,size in [('top',85),('bottom',85),('left',105),('right',105)]:
        el=OxmlElement('w:'+side); el.set(qn('w:w'),str(size)); el.set(qn('w:type'),'dxa'); margins.append(el)
    t._tbl.tblPr.append(margins)
    repeat=OxmlElement('w:tblHeader'); t.rows[0]._tr.get_or_add_trPr().append(repeat)
    for idx, values in enumerate([headers]+rows):
        row=t.rows[0] if idx==0 else t.add_row()
        cant=OxmlElement('w:cantSplit'); row._tr.get_or_add_trPr().append(cant)
        for j,(cell,value) in enumerate(zip(row.cells,values)):
            cell.width=Cm(widths[j]); cell.vertical_alignment=WD_CELL_VERTICAL_ALIGNMENT.CENTER
            shade(cell,'24465B' if idx==0 else ('F0F4F7' if idx%2==0 else 'FFFFFF'))
            p=cell.paragraphs[0]; p.paragraph_format.space_before=Pt(0); p.paragraph_format.space_after=Pt(0); p.paragraph_format.line_spacing=1.12
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER if len(str(value))<16 else WD_ALIGN_PARAGRAPH.LEFT
            r=p.add_run(str(value)); r.font.size=Pt(9.2)
            if idx==0: r.bold=True; r.font.color.rgb=RGBColor(255,255,255)
    doc.add_paragraph().paragraph_format.space_after=Pt(1)

def build(filename, title, subtitle, audience, pages):
    doc=Document(); sec=doc.sections[0]
    sec.page_width=Cm(21); sec.page_height=Cm(29.7)
    sec.top_margin=Cm(1.8); sec.bottom_margin=Cm(1.7); sec.left_margin=Cm(2); sec.right_margin=Cm(2)
    sec.header_distance=Cm(.75); sec.footer_distance=Cm(.75)
    for name,size,bold in [('Normal',10.5,False),('Title',24,True),('Subtitle',11,False),('Heading 1',15,True),('Heading 2',12,True),('Heading 3',11,True),('Caption',9,False),('Header',8,False),('Footer',8,False)]:
        set_font(doc.styles[name],size,bold)
    normal=doc.styles['Normal'].paragraph_format
    normal.line_spacing=1.22; normal.space_after=Pt(5)
    for name in ['Heading 1','Heading 2','Heading 3']:
        fmt=doc.styles[name].paragraph_format; fmt.keep_with_next=True; fmt.space_before=Pt(9); fmt.space_after=Pt(6)
    for name in ['Title','Subtitle']:
        fmt=doc.styles[name].paragraph_format; fmt.space_after=Pt(7); fmt.keep_with_next=True
        ppr=doc.styles[name].element.find(qn('w:pPr'))
        if ppr is not None:
            for b in list(ppr.findall(qn('w:pBdr'))): ppr.remove(b)
    header=sec.header.paragraphs[0]; header.text='公开 EEG 离线回放项目    '+subtitle
    footer=sec.footer.paragraphs[0]; footer.alignment=WD_ALIGN_PARAGRAPH.CENTER
    footer.add_run('第 ')
    for instruction,tail in [('PAGE',' 页  共 '),('NUMPAGES',' 页')]:
        r=footer.add_run(); fld=OxmlElement('w:fldSimple'); fld.set(qn('w:instr'),instruction)
        rr=OxmlElement('w:r'); tt=OxmlElement('w:t'); tt.text='1'; rr.append(tt); fld.append(rr); r._r.addnext(fld); footer.add_run(tail)
    props=doc.core_properties; props.title=title; props.subject=subtitle; props.author='项目文档'; props.keywords='EEG motor imagery offline public data snake MVP'
    doc.add_paragraph(title,'Title'); doc.add_paragraph(subtitle,'Subtitle')
    doc.add_paragraph('文档版本 1.0    编制日期 2026年10月20日    基于2026年10月19日模型产物')
    doc.add_paragraph('适用读者  '+audience)
    for page_idx, blocks in enumerate(pages):
        if page_idx: doc.add_page_break()
        for block in blocks:
            if block[0]=='p': doc.add_paragraph(block[1])
            elif block[0]=='h': doc.add_paragraph(block[1], 'Heading 1' if block[1][0].isdigit() else 'Heading 2')
            elif block[0]=='b':
                for item in block[1]:
                    p=doc.add_paragraph(); p.paragraph_format.left_indent=Cm(.3); p.paragraph_format.first_line_indent=Cm(-.3); p.add_run('•  '+item)
            elif block[0]=='c':
                p=doc.add_paragraph(); p.paragraph_format.left_indent=Cm(.2); p.paragraph_format.right_indent=Cm(.1); p.paragraph_format.space_before=Pt(3); p.paragraph_format.space_after=Pt(7); p.paragraph_format.line_spacing=1.05; p.paragraph_format.keep_together=True
                r=p.add_run(block[1]); r.font.name='Consolas'; r.font.size=Pt(8.5); r._element.get_or_add_rPr().rFonts.set(qn('w:eastAsia'),'Microsoft YaHei')
            elif block[0]=='t': table(doc,*block[1:])
            elif block[0]=='i':
                p=doc.add_paragraph(); p.paragraph_format.keep_with_next=True; p.add_run().add_picture(block[1],width=Cm(16.3)); doc.add_paragraph(block[2],'Caption')
    path=OUT/filename; doc.save(path)
    return path

def create_all():
    from guide_content import guide
    from proposal_content import proposal
    from prd_content import prd
    specs=[('01_EEG模型使用指南.docx','脑电左右手离线模型使用指南','Model User Guide','首次运行者 算法开发者 演示操作者',guide()),('02_Project_Proposal.docx','公开脑电驱动贪吃蛇离线演示项目','Project Proposal','项目发起者 指导与评审人员 实施团队',proposal()),('03_MVP_PRD.docx','公开脑电贪吃蛇离线演示产品需求','MVP Product Requirements Document','应用开发者 算法接口负责人 测试人员',prd())]
    result=[]
    for name,title,sub,audience,pages in specs:
        path=build(name,title,sub,audience,pages); result.append({'path':str(path),'planned_pages':len(pages)}); print(result[-1],flush=True)
    (Path(__file__).parent/'manifest.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')

if __name__=='__main__': create_all()
