#!/usr/bin/env python
# -*- coding: utf-8 -*-
import docx
import sys

doc = docx.Document('md/毕设.docx')
with open('temp_extracted.txt', 'w', encoding='utf-8') as f:
    for i, para in enumerate(doc.paragraphs[:200]):
        f.write(f"{i}: {para.text}\n")
print("Extracted to temp_extracted.txt")
