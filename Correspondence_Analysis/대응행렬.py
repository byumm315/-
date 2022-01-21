#대응행렬 만들기
from google.colab import files
myfile = files.upload()
import io
import pandas as pd
df = pd.read_excel(io.BytesIO(myfile['대응분석_Input_조선일보_감성.xlsx']))
df_1=pd.DataFrame(pd.crosstab(df.Title, df.Score, margins=False))
df_1.to_excel("대응분석_Input_조선일보.xlsx",sheet_name='sheet1',index=False)       
