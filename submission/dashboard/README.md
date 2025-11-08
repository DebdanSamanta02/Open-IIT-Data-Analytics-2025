# Team 13
## OpenIIT Dashboard

This project is a data-driven dashboard built with [Dash](https://dash.plotly.com/) designed for interactive analytics and visualization of the provided **Netflix** data. It features a sidebar navigation with seven tabs, each representing a different analytical view, and the last button supports dynamic theme switching (dark/light).

### Features
- **Sidebar Navigation:** Fixed left sidebar with tab icons for switching between views.
- **Tabs:** Seven main tabs:
  1. Executive Overview
  2. Content Intelligence
  3. Trend Analysis
  4. Geo Insights
  5. Genre Intelligence
  6. Creator Talent
  7. Strategic Recommendations
- **Theme Switcher:** Toggle between dark and light themes.
- **Modular Code:** Each tab's layout and callbacks are defined in separate files under `tabs/`.
- **Custom Styling:** Uses CSS files in `assets/` for theming and layout.

### Live Link
Check out our hosted website here: [Dashboard](https://openiit-dashboard.onrender.com)

### Project Structure

```
├── app.py                
├── requirements.txt      
├── assets/              
│   ├── dark.css
│   ├── light.css
│   ├── style.css
│   └── Tab/              
├── data/
│   └── netflix_titles.csv
├── tabs/                 
│   ├── content.py
│   ├── creator_talent.py
│   ├── exec_overview.py
│   ├── genre_intelligence.py
│   ├── geo_insights.py
│   ├── strat_recom.py
│   └── trend.py
```

## Setup
### Clone the repository
```bash
https://github.com/Jx-ls/openiit-dashboard.git
cd openiit-dashboard
```
### Create Virtual enviromnent
```bash
python -m venv venv
```
### Activate the virtual enviromnent
For Windows
```bash
venv\Scripts\activate
```
For macOS/Linux
```bash
source venv/bin/activate
```
### Install Dependancies
```bash
pip install -r requirements.txt
```
---
