cd graph-based-hs-cn

conda create --name graph-based-hs-cn python=3.8
conda activate graph-based-hs-cn

# Packages
pip install -r requirements.txt

# Special packages
git clone https://github.com/leolani/cltl-knowledgerepresentation
cd cltl-knowledgerepresentation
pip install -e .

git clone https://github.com/leolani/cltl-combot
cd cltl-combot
pip install -e .

git clone https://github.com/leolani/cltl-knowledgelinking
cd cltl-knowledgelinking
pip install -e .

git clone https://github.com/huggingface/neuralcoref.git
cd neuralcoref
pip install -r requirements.txt
pip install cython==0.29 --upgrade
pip install -e .

# final install
pip install stanford-openie
pip3 install https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-2.3.1/en_core_web_sm-2.3.1.tar.gz
