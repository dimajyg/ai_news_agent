from typing import Dict, Any, List
from lxml import etree
from .xml_schema import validate_xml

def _wc(text: str) -> int:
    return len((text or '').strip().split())

def _build_xml_document(title: str, main_idea: str, uniqueness: str, citations: List[str], links: List[str], meta: Dict[str, Any]) -> str:
    root = etree.Element('Document', source=str(meta.get('source', 'unknown')))
    if meta.get('source_name'):
        root.set('sourceName', str(meta.get('source_name')))
    if meta.get('published_date'):
        root.set('publishedDate', str(meta.get('published_date')))
    etree.SubElement(root, 'Title').text = title or ''
    etree.SubElement(root, 'MainIdea').text = main_idea or ''
    etree.SubElement(root, 'Uniqueness').text = uniqueness or ''
    rw = etree.SubElement(root, 'RelevantWorks')
    for c in citations or []:
        etree.SubElement(rw, 'Citation').text = c
    ul = etree.SubElement(root, 'UsefulLinks')
    for l in links or []:
        etree.SubElement(ul, 'Link').text = l
    xml_str = etree.tostring(root, encoding='utf-8', pretty_print=True, xml_declaration=False).decode('utf-8')
    return xml_str if validate_xml(xml_str) else ''

class BaseProcessor:
    async def process(self, item: Dict[str, Any]) -> str:
        raise NotImplementedError
