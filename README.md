# NovaRetail Customer Intelligence Dashboard

Interactive Streamlit dashboard for NovaRetail's Director of Customer Intelligence, Sophia Martinez. It shows which customers generate the most revenue, which segments show early signs of decline, and where to focus investment, across segments, regions, product categories and sales channels.

**Live app:** _add your Streamlit link here_

Challenge 2 · Team 5

## What the dashboard does

| Tab | Purpose |
|---|---|
| Overview | KPI cards, insight cards computed from the current filters, revenue and satisfaction by segment, weekly revenue trend |
| Segments & Growth Opportunities | Segment scorecard, revenue by segment and category group, average purchase by category |
| Early Warning Signs | Decline vs. all other lines, satisfaction distribution, purchase vs. satisfaction, low-satisfaction purchases |
| Region, Category & Channel | Revenue by region, channel and category (optionally split by segment), region × channel heatmap, low-satisfaction share |
| Ask the Data (AI) | Chatbot that answers questions about the data matching the current filters |
| Data Table & Download | Searchable table of the filtered data and CSV download |
| About the Data | Variables used and why, how to read the numbers, data-validation checks |

**Sidebar filters:** segment, region, product category group, retail channel, age group, gender, date range and satisfaction score, with Select All options, a Reset button and a summary of active filters.

## Data

`NR_dataset.xlsx` (course-provided): 100 purchase lines from 34 customers, January 15 to February 20, 2023. Amounts are revenue in USD.

The app cleans the data before analysis:

- One row without a segment label is shown as **Unlabeled**.
- Age group "55-64" is merged into "55+".
- 35 inconsistent product category names are grouped into 9 categories. The original name stays in the data table.
- `TransactionID` repeats across rows, so `idx` is used as the row key.
- Each row is a purchase line, not a customer. Customers are counted as unique `CustomerID`s. Some customers appear under more than one segment.

## Repository files

| File | Purpose |
|---|---|
| `app.py` | The Streamlit application (dashboard and chatbot) |
| `requirements.txt` | Python packages needed to run the app |
| `NR_dataset.xlsx` | Dataset |

## AI chatbot setup

The chatbot uses the Groq API. The API key is **not** stored in this repository. On Streamlit Community Cloud, open the app's **Settings → Secrets** and add:

```toml
GROQ_API_KEY = "your-key-here"
```

Without a key, the dashboard still works and the chatbot tab shows a setup message.

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Built with

Python, Streamlit, pandas, Plotly, openpyxl and the Groq API. The app was generated through vibe coding from a written prompt, then tested and refined.
