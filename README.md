<img width="780" height="253" alt="ACM w color" src="https://github.com/user-attachments/assets/b068a0ab-bbe7-426d-81d8-a1c70c9c2a74" />

The Discounted_Cash_Flow script is designed to perform a Discounted Cash Flow (DCF) analysis for a list of stock tickers. Here's a breakdown of its functionality:

Key Features:

Imports and Setup: Utilizes libraries like numpy, pandas, and yfinance for financial data processing and retrieval. Suppresses warnings for cleaner output.

Helper Functions: format_currency and format_percentage are utility functions to format numerical values for better readability.

Output Directories: Creates folders (dcf_reports and financial_data) to store the generated reports and financial data. 

DCF Calculation: 

The calculate_dcf function performs the following: Fetches financial data (e.g., EBIT, taxes, depreciation, etc.) and company information using the yfinance library.
Calculates Free Cash Flow to Firm (FCFF), Weighted Average Cost of Capital (WACC), and Return on Invested Capital (ROIC). Projects future FCFF values and computes the Present Value (PV) of these cash flows and the terminal value.Determines the fair value per share, upside potential, and margin of safety. Saves the analysis as a text report in the dcf_reports folder. Appends financial data to a master list for CSV export.

Error Handling: The safe_calculate_dcf function wraps the DCF calculation in a try-except block to handle errors gracefully.

Parallel Execution: Uses ThreadPoolExecutor to run the DCF analysis for multiple tickers concurrently, improving efficiency.

Ticker List: Contains a predefined list of stock tickers (e.g., "ABG.JO", "ADH.JO") for analysis.

Output:

Saves individual DCF reports for each ticker.
Compiles all financial data into a CSV file (financial_data.csv) in the financial_data folder.
Error Reporting:
Prints any errors encountered during the DCF calculation for specific tickers.

Final Messages:

Notifies the user when all reports and data have been saved.

This script is a financial analysis tool that automates the process of valuing companies based on their projected cash flows. It is particularly useful for investors and analysts looking to identify undervalued or overvalued stocks.


