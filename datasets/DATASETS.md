# Real-World Datasets

All CSV files in `datasets/raw/` are downloaded from public sources (minimum 1,500 rows each).
Run `python scripts/download_datasets.py` or `research-agent init` to fetch them.

| File | Rows | Source |
|------|------|--------|
| `telco_customer_churn.csv` | 7,043 | [IBM Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) (GitHub mirror) |
| `bank_marketing.csv` | 41,188 | [UCI Bank Marketing](https://archive.ics.uci.edu/dataset/222/bank+marketing) |
| `california_housing.csv` | 20,640 | [California Housing (1990 census)](https://github.com/ageron/handson-ml2/tree/master/datasets/housing) |
| `adult_income.csv` | 32,561 | [UCI Adult (Census Income)](https://archive.ics.uci.edu/dataset/2/adult) |
| `credit_card_default.csv` | 30,000 | [UCI Default of Credit Card Clients](https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients) |
| `online_shoppers.csv` | 12,330 | [UCI Online Shoppers Purchasing Intention](https://archive.ics.uci.edu/dataset/468/online+shoppers+purchasing+intention+dataset) |

## Download manually (optional)

If the script fails, download from the links above and place files in `datasets/raw/` using the filenames in the table.

## Benchmark tasks

Task definitions mapping queries to these datasets live in `datasets/benchmark_tasks/tasks.json` (25 tasks, easy → hard).
