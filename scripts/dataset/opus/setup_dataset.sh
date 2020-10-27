
. ./const.sh
script_dir=$(cd $(dirname $0); pwd)
opus_data_root=$(pwd)/$dataset_root/opus.en-de

# Download datasets

# Law (JRC-Acquis) 
if [ ! -e $opus_data_root/law ]; then
    mkdir -p $opus_data_root/law/original
    cd $opus_data_root/law/original
    wget https://object.pouta.csc.fi/OPUS-JRC-Acquis/v3.0/moses/de-en.txt.zip -O JRC-Acquis.zip
    unzip JRC-Acquis.zip
    cd $opus_data_root
fi 

# IT (GNOME, KDE, PHP, Ubuntu, and OpenOffice)
if [ ! -e $opus_data_root/IT ]; then
    mkdir -p $opus_data_root/IT/original
    cd $opus_data_root/IT/original
    wget https://object.pouta.csc.fi/OPUS-GNOME/v1/moses/de-en.txt.zip -O GNOME.zip
    wget https://object.pouta.csc.fi/OPUS-KDE4/v2/moses/de-en.txt.zip -O KDE4.zip
    wget https://object.pouta.csc.fi/OPUS-PHP/v1/moses/de-en.txt.zip -O PHP.zip
    wget https://object.pouta.csc.fi/OPUS-Ubuntu/v14.10/moses/de-en.txt.zip -O Ubuntu.zip
    unzip -o GNOME.zip
    unzip -o KDE4.zip
    unzip -o PHP.zip
    unzip -o Ubuntu.zip
    cd $opus_data_root
fi 

#  Medical (EMEA)
if [ ! -e $opus_data_root/medical ]; then
    mkdir -p $opus_data_root/medical/original
    cd $opus_data_root/medical/original
    wget https://object.pouta.csc.fi/OPUS-EMEA/v3/moses/de-en.txt.zip -O EMEA.zip
    unzip EMEA.zip
    cd $opus_data_root
fi 

# Koran (Tanzil)
if [ ! -e $opus_data_root/koran ]; then
    mkdir -p $opus_data_root/koran/original
    cd $opus_data_root/koran/original
    wget https://object.pouta.csc.fi/OPUS-Tanzil/v1/moses/de-en.txt.zip -O Tanzil.zip
    unzip Tanzil.zip
    cd $opus_data_root
fi 

# Subtitles (Opensubtitle2018)
if [ ! -e $opus_data_root/subtitles ]; then
    mkdir -p $opus_data_root/subtitles/original
    cd $opus_data_root/subtitles/original
    wget https://object.pouta.csc.fi/OPUS-OpenSubtitles/v2018/moses/de-en.txt.zip -O OpenSubtitles.zip
    unzip OpenSubtitles.zip
    cd $opus_data_root
fi 

# (TODO)
# 


# # Law (JRC-Acquis) 
# https://object.pouta.csc.fi/OPUS-JRC-Acquis/v3.0/moses/de-en.txt.zip -O JRC-Acquis.zip

# # IT (GNOME, KDE, PHP, Ubuntu, and OpenOffice)
# https://object.pouta.csc.fi/OPUS-GNOME/v1/moses/de-en.txt.zip
# https://object.pouta.csc.fi/OPUS-KDE4/v2/moses/de-en.txt.zip
# https://object.pouta.csc.fi/OPUS-PHP/v1/moses/de-en.txt.zip
# https://object.pouta.csc.fi/OPUS-Ubuntu/v14.10/moses/de-en.txt.zip
# http://opus.nlpl.eu/download.php?f=OpenOffice/v3/moses/de-en_GB.txt.zip # OpenOffice?

# # Medical (EMEA)
# https://object.pouta.csc.fi/OPUS-EMEA/v3/moses/de-en.txt.zip

# # Koran (Tanzil)
# https://object.pouta.csc.fi/OPUS-Tanzil/v1/moses/de-en.txt.zip

# # Subtitles (OpenSubtitles)
# https://object.pouta.csc.fi/OPUS-OpenSubtitles/v2018/moses/de-en.txt.zip


