import pdfplumber
import pandas as pd
# import fitz 

class Resume:
    def __init__(self, source, is_pdf=True):
        self.source = source
        
        if is_pdf:
            # if read from pdf
            self.text = self._extract_text_from_pdf()
        else:
            # read direct from text
            self.text = source
        return
    
    def _extract_text_from_pdf(self):
        with pdfplumber.open(self.source) as pdf:
            text = ""
            for page in pdf.pages:
                text += page.extract_text() or ""
        return text
    
        # doc = fitz.open(self.source)
        # text = ""
        # for page in doc:
        #     text += page.get_text()
        # return text


def main():
    df = pd.read_csv('datasets/UpdatedResumeDataSet.csv')
    source = df.iloc[0]['Resume']

    resume = Resume(source=source)
    print(resume.text)
    return

if __name__ == '__main__':
    main()
