f='notes'

pdflatex $f.tex
rm -rf   $f.aux $f.log $f.synctex.gz
