from lxml import etree

XML_SCHEMA = '''
<xs:schema xmlns:xs="http://www.w3.org/2001/XMLSchema" elementFormDefault="qualified">
  <xs:element name="Document">
    <xs:complexType>
      <xs:sequence>
        <xs:element name="Title" type="xs:string"/>
        <xs:element name="MainIdea" type="xs:string"/>
        <xs:element name="Uniqueness" type="xs:string"/>
        <xs:element name="FurtherResearch">
          <xs:complexType>
            <xs:sequence>
              <xs:element name="Idea" type="xs:string" minOccurs="0" maxOccurs="unbounded"/>
            </xs:sequence>
          </xs:complexType>
        </xs:element>
        <xs:element name="RelevantWorks">
          <xs:complexType>
            <xs:sequence>
              <xs:element name="Citation" type="xs:string" minOccurs="0" maxOccurs="unbounded"/>
            </xs:sequence>
          </xs:complexType>
        </xs:element>
        <xs:element name="UsefulLinks">
          <xs:complexType>
            <xs:sequence>
              <xs:element name="Link" type="xs:anyURI" minOccurs="0" maxOccurs="unbounded"/>
            </xs:sequence>
          </xs:complexType>
        </xs:element>
      </xs:sequence>
      <xs:attribute name="source" type="xs:string" use="required"/>
      <xs:attribute name="sourceName" type="xs:string" use="optional"/>
      <xs:attribute name="publishedDate" type="xs:string" use="optional"/>
    </xs:complexType>
  </xs:element>
</xs:schema>
'''

def get_schema():
    parser = etree.XMLParser()
    xmlschema_doc = etree.fromstring(XML_SCHEMA.encode('utf-8'), parser)
    return etree.XMLSchema(xmlschema_doc)

def validate_xml(xml_str: str) -> bool:
    schema = get_schema()
    try:
        doc = etree.fromstring(xml_str.encode('utf-8'))
        return schema.validate(doc)
    except Exception:
        return False

def validate_xml_detailed(xml_str: str):
    """Return (is_valid, errors) using lxml's error log for better debugging."""
    schema = get_schema()
    errors = []
    try:
        doc = etree.fromstring(xml_str.encode('utf-8'))
        is_valid = schema.validate(doc)
        if not is_valid:
            for e in schema.error_log:
                errors.append(f"line {e.line}, column {e.column}: {e.message}")
        return is_valid, errors
    except Exception as ex:
        return False, [str(ex)]
