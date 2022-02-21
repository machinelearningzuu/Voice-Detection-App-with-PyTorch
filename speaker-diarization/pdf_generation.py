from docxtpl import DocxTemplate
from datetime import datetime
from docx2pdf import convert
import os

from variables import *

# tbl_contents = [
#                 {'label': 'SPEAKER 1', 'cols': ['20', '40', '20']},
#                 {'label': 'SPEAKER 2', 'cols': ['8', '14', '6']},
#                 {'label': 'SPEAKER 3', 'cols': ['5', '10', '5']},
#                 {'label': 'SPEAKER 1', 'cols': ['1', '4', '3']}
#                 ]

# tb2_contents = [
#                 {'label': 'SPEAKER 1', 'percentage': '20%'},
#                 {'label': 'SPEAKER 2', 'percentage': '40%'},
#                 {'label': 'SPEAKER 3', 'percentage': '40%'}
#                 ]

def format_speaker_outputs(speakerdf):
    tbl_contents = []
    speakerdf = speakerdf[['SpeakerLabel', 'StartTime', 'EndTime', 'TimeSeconds']]
    for i in range(len(speakerdf)):
        tbl_content_i = {
                      'label': speakerdf.loc[i, 'SpeakerLabel'],  
                      'cols': [
                            int(speakerdf.loc[i, 'StartTime']), 
                            int(speakerdf.loc[i, 'EndTime']), 
                            int(speakerdf.loc[i, 'TimeSeconds'])
                            ]
                      }
        tbl_contents.append(tbl_content_i)

    return tbl_contents

def format_speaker_summary(summarydf):
    tb2_contents = []
    for i in range(len(summarydf)):
        tb2_content_i = {
                      'label': summarydf.loc[i, 'SpeakerLabel'],  
                      'percentage': summarydf.loc[i, 'Percentage']
                      }
        tb2_contents.append(tb2_content_i)

    return tb2_contents

def pdfGeneration(
            tbl_contents, 
            tb2_contents,
            file_name
                 ):
    doc = DocxTemplate(template_path)
    now = datetime.now()
    Date= now.strftime("%H:%M:%S")
    Time= now.strftime("%d/%m/%Y")

    context = {'Date':Date,
              'Time':Time,
              'Name' : student, 
              'tbl_contents':tbl_contents,
              'tb2_contents':tb2_contents
              }


    now = datetime.now()

    if not os.path.exists(output_dir.format(file_name.split('.')[0])):
        os.makedirs(output_dir.format(file_name.split('.')[0]))

    doc_path = os.path.join(output_dir.format(file_name.split('.')[0]), 'report.docx')
    pdf_path = os.path.join(output_dir.format(file_name.split('.')[0]), 'report.pdf')

    doc.render(context)
    doc.save(doc_path)
    convert(doc_path, pdf_path)