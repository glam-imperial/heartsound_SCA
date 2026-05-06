# Prior-Aligned Frequency-Domain Explanations for Heart Sound Classification  
## A Scale-Consistent Attribution Approach

This repository is the **official implementation** of the paper:

> **Prior-Aligned Frequency-Domain Explanations for Heart Sound Classification: A Scale-Consistent Attribution Approach**

---

## 📌 File Descriptions

- **`script/`**  
  Contains the core implementation of the proposed method.

- **`data/`**  
  Stores the datasets used in the experiments.

- **`heart1.py` / `heart2.py`**  
  Main experiment scripts:
  - `heart1.py`: Experiments on the **PhysioNet 2016 dataset**
  - `heart2.py`: Experiments on the **circor 2022 dataset**

- **`ablation1.py`/`ablation2.py`**  
  Code for ablation studies to evaluate different components of the method.

- **`stat_test.py`/**  
  Scripts for statistical significance testing of experimental results.

- **`analyze_sca_prior_alignment1.py`/`analyze_sca_prior_alignment2.py`**  
  Tools for analyzing **scale-consistent attribution** and **prior alignment**.

---

## 📖 Citation

If you find this work useful, please cite:

```bibtex
@article{sun9prior,
  title={Prior-Aligned Frequency-Domain Explanations for Heart Sound Classification: A Scale-Consistent Attribution Approach},
  author={Sun, Qiyang and Li, Yupei and Javadov, Aydin and Wu, Xiaoliang and Schuller, Bj{\"o}rn Wolfgang},
  journal={Frontiers in Artificial Intelligence},
  volume={9},
  pages={1780496},
  publisher={Frontiers}
}