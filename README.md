download data for the script using 
curl -L -o ./smart-meter-data-mathura-and-bareilly.zip https://www.kaggle.com/api/v1/datasets/download/jehanbhathena/smart-meter-data-mathura-and-bareilly

unzip it in the folder
run the following commands

python process_data.py CEEW\ -\ Smart\ meter\ data\ Bareilly\ 2020.csv Bareilly_2020_15t.csv

python process_data.py CEEW\ -\ Smart\ meter\ data\ Bareilly\ 2021.csv Bareilly_2021_15t.csv

python process_data.py CEEW\ -\ Smart\ meter\ data\ Mathura\ 2020.csv Mathura_2020_15t.csv

python process_data.py SM\ Cleaned\ Data\ MH2021.csv Mathura_2021_15t.csv


After running above commands, to run prediction

python ta_predictor.py <dataset> <to_open_plot>

where <dataset> can be b for Bareilly, m for Mathura
<to_open_plot> can be p to open each plot in gui, else n to just save it to directory
