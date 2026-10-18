# EcoCart AI: Multi-Agent Logistics Pathfinding & Retail Demand Forecasting

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![NetworkX](https://img.shields.io/badge/NetworkX-Graph_Optimization-blue.svg)](https://networkx.org/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.3%2B-orange.svg)](https://scikit-learn.org/)
[![National College of Ireland](https://img.shields.io/badge/Institution-National_College_of_Ireland-blue.svg)](https://www.ncirl.ie/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

An intelligent logistics and e-commerce optimization system combining heuristic search pathfinding with machine learning retail demand forecasting and algorithmic fairness audits.

---

## 📌 Project Overview

EcoCart is an e-commerce platform addressing sustainable delivery routing, inventory optimization, and equitable customer engagement. This project develops:
1. **Multi-Agent Logistics Architecture**: Goal-based and utility-based agents balancing transit time, distance, vehicle capacity, and carbon emissions.
2. **Heuristic Search Routing Engine**: Comparative algorithmic evaluation across urban and rural logistics topologies.
3. **Machine Learning Demand Forecasting**: Regression models predicting product sales demand from price, promotions, and web traffic.
4. **Algorithmic Fairness Audit**: Framework to identify and mitigate marketing biases against rural consumer demographics.
5. **Interactive Web Routing Prototype**: HTML5 simulation visualizing node exploration and optimal path selection.

---

## 🔬 Search Algorithms Comparative Benchmark

The routing engine benchmarks four classic and heuristic graph search algorithms on weighted delivery networks:

| Algorithm | Routing Strategy | Urban Nodes Visited | Rural Nodes Visited | Optimality | Memory Efficiency |
|---|---|---|---|---|---|
| **Depth-First Search (DFS)** | Blind Depth Traversal | 6 | 12 | Non-Optimal | High |
| **Breadth-First Search (BFS)** | Level-by-Level Blind | 13 | 28 | Shortest Hop Only | Low |
| **A\* Search Algorithm** | $f(n) = g(n) + h(n)$ | **5** | **7** | **Strictly Optimal (Cost: 12 Urban / 31 Rural)** | Medium |
| **Iterative Deepening A\* (IDA\*)** | Depth-Bounded Heuristic | 17 | 44 | Optimal (Cost: 12 Urban / 31 Rural) | **Optimal (O(bd))** |

### Key Finding:
- **A\* Search** is optimal for route execution speed, visiting up to **84% fewer states** than IDA\* in sparse rural topologies while maintaining minimal path cost.
- **IDA\*** minimizes memory footprints on constrained embedded edge devices.

---

## 📈 Demand Forecasting Performance

Supervised regression models trained on historical retail sales indicators:

| Machine Learning Model | Mean Absolute Error (MAE) | Root Mean Squared Error (RMSE) | $R^2$ Score |
|---|---|---|---|
| **Linear Regression** | **18.5989** | **23.4980** | **0.9488** |
| Random Forest Regressor | 27.4200 | 37.5752 | 0.8812 |
| Decision Tree Regressor | 38.1500 | 51.2831 | 0.7760 |

---

## 📁 Repository Structure

```
.
├── Code/
│   ├── question_3_route_optimization.py       # DFS, BFS, A* routing benchmark
│   ├── question_4_astar_idastar_comparison.py  # Urban vs Rural A* & IDA* comparison
│   ├── question_5_demand_forecasting.py       # ML demand forecasting models
│   └── requirements.txt
├── Dataset/
│   └── Q5_sampledataset.csv                   # Historical retail demand dataset
├── Graphs/
│   ├── Question 3-plot1.png ... Question 5-plot2.png
├── Prototype/
│   └── question_3_prototype.html              # Interactive browser pathfinding simulator
├── requirements.txt
├── .gitignore
└── README.md
```

---

## 🚀 Quick Start

### Installation
```bash
git clone https://github.com/Bhavitha025/ecocart-logistics-ai.git
cd ecocart-logistics-ai
pip install -r requirements.txt
```

### Running Routing & ML Scripts
- **Route Optimization Benchmark**:
  ```bash
  python Code/question_3_route_optimization.py
  ```
- **A\* vs IDA\* Comparative Simulation**:
  ```bash
  python Code/question_4_astar_idastar_comparison.py
  ```
- **Demand Forecasting Engine**:
  ```bash
  python Code/question_5_demand_forecasting.py
  ```
- **Interactive Routing Visualizer**:
  Open `Prototype/question_3_prototype.html` directly in your browser.

---

## 📜 License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
