/***SAS 코드입니다.***/

options validvarname=any;
DATA data;
infile "C:\Users\tyumi\Documents\논문데이터\Bing\대응분석_Input_조선일보.csv" dlm=',' firstobs=2;
Length title $10 긍정 8 부정 8 중립 8 ;
input title 긍정 부정 중립;
run;

proc corresp data=data profile=row cellchi2 cp expected observed all;
var _numeric_;
id title;
run;
