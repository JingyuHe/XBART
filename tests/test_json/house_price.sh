echo "Testing JSON capabilities"

echo "Building Model in python..."
python house_price.py

echo "Loading and predicting in R"
Rscript house_price.R