# CBP_DR_Opt
北京科技大学AI+叶片高效优化团队-《基于非线性降维的压气机叶型优化方法研究》-优化体系——作者：李世荣

首先感谢我的毕设导师——成金鑫老师，从基础到创新到完成毕业设计，他一路为我指引正确的方向，并始终耐心解答我的疑问，他提供了融合边界自适应拓展的贝叶斯优化进化优化算法和很多研究方法以及思路。其次感谢北京航空航天大学黄鹏飞学长提供的基于pca降维的自适应扩展优化体系，他为我开辟了框架，并指导我入门。感谢杨俊逸同学提供的原始双圆弧参数化叶型库，感谢钟镇强同学提供的原始椭圆参数化叶型库。

Blade_optimization文件夹中含完整基于pca和kpca降维的边界自适应放缩优化体系应该包含的文件夹

其中特别感谢北京航空航天大学宁方飞教授团队提供的MAP-S1流场计算软件，并在邮件中对我的指导，但因为版权原因，本开源库不提供该软件

<img width="548" alt="完整优化体系图" src="https://github.com/user-attachments/assets/a627de68-4a94-470e-b380-fc2307786457" />

Nonlinear_measurement文件夹中含检测叶型库“三维主成分体线比”，“重构相似度”，“重构非线性度”三个非线性度指标的计算程序

Dimple_detection文件夹中含检测叶型库中缺陷叶型的程序，单个叶型缺陷检测程序已经包含在基于pca和kpca降维的边界自适应放缩优化体系中。

Optresults_analyze文件夹中含对优化结果进行分析的程序，其中Opt_leaf_comparison实现叶型对比分析，Opt_spre_comparison实现叶型表面静压对比分析

CBP_library含研究过程所使用的压气机叶型库，以及各种变换算法。

讲解视频链接：

夸克网盘「AI+高效优化-《基于非线性降维的压气机叶型优化方法研究》」
链接：https://pan.quark.cn/s/67f03095b997
提取码：bNKp

视频目录：

<img width="553" alt="9caf5223cf51e122e7cbad05446812f" src="https://github.com/user-attachments/assets/6883c0b8-f206-40b0-8bf4-b50af240e3c2" />
好的，我已根据您提供的README内容，为您创建了一个专业、清晰的英文版本。这个版本保留了原文的所有信息点和感谢，并按照国际开源项目的通用格式进行了组织。

---
Translation

# CBP_DR_Opt

**AI-Enhanced High-Efficiency Blade Optimization Research @ University of Science and Technology Beijing**

**Topic:** *A Dimensionality-Reduction-Based Optimization Method for Compressor Blade Profiles in Nonlinear Design Spaces*

**Author:** Li Shirong

---

## Overview

This repository hosts the complete optimization framework developed for my undergraduate thesis, which focuses on an innovative dimensionality-reduction-based optimization method for compressor blade profiles in nonlinear design spaces.

The core of this work is a collaborative optimization system that integrates **PCA/KPCA dimensionality reduction** with an **adaptive boundary scaling Bayesian optimization algorithm**.

## Acknowledgements

I would like to express my sincere gratitude to the following individuals and teams:

*   **My thesis supervisor, Mr. Cheng Jinxin:** He has been an invaluable guide throughout this journey—from building foundational knowledge to driving innovation and completing the thesis. He provided the adaptive boundary scaling Bayesian optimization algorithm, numerous research methodologies, and思路, and patiently addressed all my questions.
*   **Huang Pengfei (Senior from Beihang University):** He provided the initial adaptive expansion optimization framework based on PCA dimensionality reduction, paved the way for this project, and guided me through the initial stages.
*   **Yang Junyi:** For providing the original double-circular-arc parameterized blade profile library.
*   **Zhong Zhenqiang:** For providing the original ellipse parameterized blade profile library.
*   **The team of Professor Ning Fangfei at Beihang University:** For providing the MAP-S1 flow field calculation software and their guidance via email.

> **Note Regarding MAP-S1 Software:** Due to copyright restrictions, the MAP-S1 flow field calculation software is **not included** in this repository.

## Repository Structure

*   `Blade_optimization/`
    Contains the **complete adaptive boundary scaling optimization system** based on PCA and KPCA dimensionality reduction. This includes all necessary folders for a functioning setup.
    *   *Includes a single blade defect detection program.*

*   `Nonlinear_measurement/`
    Contains programs for calculating three nonlinearity metrics for blade profile libraries:
    1.  3D Principal Component Volume-to-Line Ratio
    2.  Reconstruction Similarity
    3.  Reconstruction Nonlinearity (Γ)

*   `Dimple_detection/`
    Contains programs for detecting defective blade profiles within a library.

*   `Optresults_analyze/`
    Contains programs for analyzing optimization results.
    *   `Opt_leaf_comparison`: For blade profile comparison analysis.
    *   `Opt_spre_comparison`: For blade surface static pressure comparison analysis.

*   `CBP_library/`
    Contains the compressor blade profile libraries used in the research, along with various transformation algorithms.

## Framework Diagram

The following diagram illustrates the complete optimization framework:

<img width="548" alt="Complete Optimization Framework Diagram" src="https://github.com/user-attachments/assets/a627de68-4a94-470e-b380-fc2307786457" />

## Explanation Video (in Chinese)

A detailed video explanation (in Chinese) is available via Quark Netdisk:

*   **Link:** [https://pan.quark.cn/s/67f03095b997](https://pan.quark.cn/s/67f03095b997)
*   **Extraction Code:** `bNKp`

### Video Content Index

<img width="553" alt="Video Content Index" src="https://github.com/user-attachments/assets/6883c0b8-f206-40b0-8bf4-b50af240e3c2" />

---

希望这个英文版本能帮助您的项目获得更广泛的国际关注！如果需要对某些部分的措辞进行调整，请随时告诉我。

