# LOW-RANK ADAPTATION (LORA) FOR ALTERNATIVE DATA CREDIT SCORING: ENHANCING FINANCIAL INCLUSION UNDER THE NATIONAL DEVELOPMENT STRATEGY

by

**PAULA MARISA**  
H240799Q

A dissertation submitted in partial fulfilment of the requirements for the degree of

**MASTER OF TECHNOLOGY IN SOFTWARE ENGINEERING**

in the

**DEPARTMENT OF COMPUTER SCIENCE**  
FACULTY OF INFORMATION SCIENCES AND ENGINEERING

at the

**HARARE INSTITUTE OF TECHNOLOGY**

March 2026

---

# DECLARATION

I, Paula Marisa (H240799Q), hereby declare that this dissertation is my own original work and that all sources used or quoted have been indicated and acknowledged by means of complete references. This work has not been submitted before for any other degree or examination at any other institution.

**Paula Marisa**  
Date: _______________

---

# SUPERVISOR'S CERTIFICATE

This is to certify that this dissertation has been submitted with my approval as the university supervisor:

**Supervisor's Name:** _______________  
Date: _______________

---

# ABSTRACT

Financial exclusion remains a critical challenge in Zimbabwe, with 89% of the population lacking access to formal credit services. Traditional credit scoring systems rely heavily on conventional financial histories, systematically excluding the unbanked and underbanked populations who lack formal banking relationships. This research investigates the application of Low-Rank Adaptation (LoRA), a parameter-efficient fine-tuning technique, to develop an alternative data credit scoring system that addresses this gap.

The study leverages diverse alternative data sources including mobile money transaction patterns, utility payment histories, and digital commerce activities to construct comprehensive creditworthiness profiles. By implementing LoRA's low-rank decomposition approach, the research achieves significant computational efficiency while maintaining model performance, reducing trainable parameters by 99.7% compared to full fine-tuning approaches.

This research aligns with Zimbabwe's National Development Strategy (NDS1 2021-2025, NDS2 2026-2030) and the newly launched National AI Strategy (2026-2030), contributing to the national goal of achieving over 90% financial inclusion by 2030. The proposed system addresses the USD 1.5 billion MSME financing gap while promoting responsible AI deployment in financial services.

The dissertation presents a comprehensive three-phase methodology encompassing synthetic data generation, LoRA-enhanced model development, and multi-dimensional evaluation across predictive performance, fairness, and computational efficiency metrics. Expected results demonstrate the feasibility of deploying sophisticated machine learning models in resource-constrained environments while maintaining ethical standards and regulatory compliance.

**Keywords:** Low-Rank Adaptation, Alternative Data, Credit Scoring, Financial Inclusion, Parameter-Efficient Fine-Tuning, Zimbabwe, Machine Learning, FinTech

---

# LIST OF ABBREVIATIONS

| Abbreviation | Full Form |
|--------------|-----------|
| AI | Artificial Intelligence |
| API | Application Programming Interface |
| AUC | Area Under Curve |
| BERT | Bidirectional Encoder Representations from Transformers |
| DIR | Disparate Impact Ratio |
| EOD | Equalized Odds Difference |
| FPR | False Positive Rate |
| GPU | Graphics Processing Unit |
| KYC | Know Your Customer |
| LoRA | Low-Rank Adaptation |
| LSTM | Long Short-Term Memory |
| ML | Machine Learning |
| MSME | Micro, Small, and Medium Enterprises |
| NDS | National Development Strategy |
| NLP | Natural Language Processing |
| PEFT | Parameter-Efficient Fine-Tuning |
| ROC | Receiver Operating Characteristic |
| SMOTE | Synthetic Minority Over-sampling Technique |
| TPR | True Positive Rate |
| USD | United States Dollar |
| XAI | Explainable Artificial Intelligence |

---

# TABLE OF CONTENTS

1. [CHAPTER 1: INTRODUCTION AND BACKGROUND TO THE STUDY](#chapter-1-introduction-and-background-to-the-study)
   - 1.1 [Introduction](#11-introduction)
   - 1.2 [Background to the Study](#12-background-to-the-study)
   - 1.3 [Problem Statement](#13-problem-statement)
   - 1.4 [Research Aim and Objectives](#14-research-aim-and-objectives)
   - 1.5 [Research Questions](#15-research-questions)
   - 1.6 [Significance of the Study](#16-significance-of-the-study)
   - 1.7 [Scope and Limitations](#17-scope-and-limitations)
   - 1.8 [Dissertation Structure](#18-dissertation-structure)

2. [CHAPTER 2: LITERATURE REVIEW](#chapter-2-literature-review)
   - 2.1 [Introduction](#21-introduction)
   - 2.2 [Theoretical Framework](#22-theoretical-framework)
   - 2.3 [Traditional Credit Scoring Systems](#23-traditional-credit-scoring-systems)
   - 2.4 [Alternative Data in Credit Assessment](#24-alternative-data-in-credit-assessment)
   - 2.5 [Machine Learning in Credit Scoring](#25-machine-learning-in-credit-scoring)
   - 2.6 [Parameter-Efficient Fine-Tuning and LoRA](#26-parameter-efficient-fine-tuning-and-lora)
   - 2.7 [Financial Inclusion in Developing Economies](#27-financial-inclusion-in-developing-economies)
   - 2.8 [Ethical and Regulatory Considerations](#28-ethical-and-regulatory-considerations)
   - 2.9 [Research Gaps](#29-research-gaps)
   - 2.10 [Chapter Summary](#210-chapter-summary)

3. [CHAPTER 3: RESEARCH METHODOLOGY](#chapter-3-research-methodology)
   - 3.1 [Introduction](#31-introduction)
   - 3.2 [Research Philosophy and Design](#32-research-philosophy-and-design)
   - 3.3 [Data Collection and Preparation](#33-data-collection-and-preparation)
   - 3.4 [Model Development](#34-model-development)
   - 3.5 [Evaluation Framework](#35-evaluation-framework)
   - 3.6 [Ethical Considerations](#36-ethical-considerations)
   - 3.7 [Chapter Summary](#37-chapter-summary)

4. [REFERENCES](#references)

---

# CHAPTER 1: INTRODUCTION AND BACKGROUND TO THE STUDY

## 1.1 Introduction

Access to formal credit remains one of the most significant barriers to economic participation and poverty reduction in developing economies. In Zimbabwe, despite remarkable progress in overall financial inclusion—rising from 69% in 2014 to 84% in 2024—an estimated 89% of the population still lacks access to formal credit services (FinMark Trust, 2024). This paradox highlights a critical gap: while individuals may have access to basic financial services such as mobile money accounts, they remain excluded from credit facilities that could enable entrepreneurship, asset accumulation, and economic mobility.

Traditional credit scoring systems, exemplified by models such as FICO scores, rely predominantly on conventional financial histories including bank account activity, loan repayment records, and credit card usage patterns. These systems systematically exclude the unbanked and underbanked populations who lack formal banking relationships, creating a self-reinforcing cycle of financial exclusion. In Zimbabwe's context, where informal economic activities account for approximately 60% of GDP and mobile money transactions exceeded USD 40 billion in 2023, conventional credit assessment methodologies fail to capture the creditworthiness signals embedded in alternative data sources (Reserve Bank of Zimbabwe, 2024).

The emergence of alternative data—encompassing mobile money transaction patterns, utility payment histories, digital commerce activities, and social network behaviours—presents unprecedented opportunities to construct more inclusive creditworthiness profiles. However, the computational complexity and resource requirements of sophisticated machine learning models capable of processing such diverse data sources pose significant challenges for deployment in resource-constrained environments characteristic of developing economies.

This research investigates the application of Low-Rank Adaptation (LoRA), a parameter-efficient fine-tuning technique, to develop an alternative data credit scoring system tailored to Zimbabwe's unique socio-economic context. LoRA's ability to reduce trainable parameters by up to 99.7% while maintaining model performance makes it particularly suitable for deployment in environments with limited computational infrastructure (Hu et al., 2021). By leveraging LoRA's low-rank decomposition approach, this study aims to demonstrate that sophisticated AI-driven credit assessment can be both technically feasible and economically viable in developing economy contexts.

This research aligns strategically with Zimbabwe's National Development Strategy (NDS1 2021-2025, NDS2 2026-2030) and the newly launched National AI Strategy (2026-2030), which collectively prioritise financial inclusion, digital transformation, and responsible AI deployment as key enablers of sustainable economic development. The study contributes to the national goal of achieving over 90% financial inclusion by 2030 while addressing the USD 1.5 billion financing gap faced by Micro, Small, and Medium Enterprises (MSMEs) (Ministry of Finance and Economic Development, 2025).

## 1.2 Background to the Study

### 1.2.1 The Financial Inclusion Challenge in Zimbabwe

Zimbabwe's financial landscape has undergone significant transformation over the past decade, driven primarily by mobile money adoption and digital payment innovations. The FinScope Consumer Survey 2024 reveals that 84% of adults now have access to formal financial services, representing a substantial increase from 69% in 2014 (FinMark Trust, 2024). However, this aggregate figure masks persistent disparities in credit access, with only 11% of adults having access to formal credit facilities.

The credit access gap is particularly pronounced among specific demographic segments. Rural populations, women entrepreneurs, youth, and informal sector workers face disproportionate barriers to credit access, perpetuating cycles of poverty and limiting economic opportunities. The World Bank estimates that Zimbabwe's MSME sector, which employs over 5 million people, faces a financing gap of approximately USD 1.5 billion, constraining business growth and job creation (World Bank, 2023).

**[Figure 1.1: Financial Inclusion Trends in Zimbabwe (2014-2024)]**  
*Caption: Evolution of financial inclusion metrics showing the gap between overall financial access (84%) and formal credit access (11%). Data source: FinScope Consumer Survey 2024.*

### 1.2.2 Limitations of Traditional Credit Scoring

Traditional credit scoring methodologies, developed primarily in advanced economies with mature financial systems, rely on structured credit bureau data including loan repayment histories, credit card utilisation, and bank account behaviours. The FICO score, introduced in 1989 and widely adopted globally, exemplifies this approach by synthesising payment history (35%), amounts owed (30%), length of credit history (15%), new credit (10%), and credit mix (10%) into a single numerical score (Fair Isaac Corporation, 2020).

However, these methodologies encounter fundamental limitations in developing economy contexts. In Zimbabwe, credit bureau coverage remains limited, with only 15% of adults having credit bureau records (Reserve Bank of Zimbabwe, 2024). The majority of the population, engaged primarily in informal economic activities and relying on cash or mobile money transactions, remains invisible to traditional credit assessment systems. This "data poverty" creates a paradox: those most in need of credit to escape poverty are systematically excluded due to lack of conventional credit histories.

**[Table 1.1: Comparison of Traditional vs. Alternative Data Sources for Credit Scoring]**

| **Data Category** | **Traditional Sources** | **Alternative Sources** | **Availability in Zimbabwe** |
|-------------------|------------------------|------------------------|------------------------------|
| Transaction History | Bank statements, credit card records | Mobile money transactions, digital payments | High (mobile money) |
| Payment Behaviour | Loan repayment records | Utility payments, rent payments, airtime purchases | Moderate to High |
| Financial Stability | Savings account balances | Digital wallet balances, remittance patterns | High |
| Social Capital | Credit references | Social network data, community vouching | Moderate |
| Economic Activity | Employment records | Digital commerce, gig economy participation | Growing |

*Caption: Comparative analysis of data sources highlighting the superior availability of alternative data in Zimbabwe's context.*

### 1.2.3 The Rise of Alternative Data

Alternative data encompasses non-traditional information sources that can provide insights into an individual's creditworthiness. These include mobile money transaction patterns, utility payment histories, digital commerce activities, telecommunications usage, social media behaviours, and psychometric assessments. Research demonstrates that alternative data can significantly enhance credit risk prediction, particularly for thin-file and no-file populations (Berg et al., 2020).

In Zimbabwe, the proliferation of mobile money platforms—led by EcoCash, which processes over 90% of mobile money transactions—has generated vast datasets capturing financial behaviours of previously unbanked populations. In 2023, mobile money transactions exceeded USD 40 billion, representing approximately 150% of GDP (Reserve Bank of Zimbabwe, 2024). These transaction patterns, encompassing peer-to-peer transfers, merchant payments, utility bill settlements, and savings behaviours, offer rich signals of financial discipline and creditworthiness.

Similarly, utility payment data from electricity providers (ZESA), water utilities, and telecommunications companies provide longitudinal records of payment consistency and financial management capabilities. Digital commerce platforms and gig economy participation further contribute to comprehensive financial behaviour profiles.

### 1.2.4 Machine Learning and Parameter-Efficient Fine-Tuning

The application of machine learning to credit scoring has evolved significantly over the past two decades. Early approaches utilised logistic regression and decision trees, while contemporary systems employ sophisticated neural networks, ensemble methods, and deep learning architectures capable of capturing complex non-linear relationships in high-dimensional data (Khandani et al., 2010).

However, the deployment of large-scale machine learning models, particularly transformer-based architectures that have revolutionised natural language processing and sequential data analysis, presents significant computational challenges. Full fine-tuning of models such as BERT or GPT requires substantial computational resources, extensive training time, and significant energy consumption—barriers that are particularly acute in resource-constrained developing economy contexts.

Parameter-Efficient Fine-Tuning (PEFT) techniques have emerged as a solution to these challenges. LoRA, introduced by Hu et al. (2021), represents a breakthrough in PEFT by decomposing weight updates into low-rank matrices, reducing trainable parameters by up to 99.7% while maintaining or even improving model performance. This efficiency makes LoRA particularly suitable for deployment in environments with limited computational infrastructure.

**[Figure 1.2: LoRA Architecture and Low-Rank Decomposition]**  
*Caption: Illustration of LoRA's low-rank decomposition approach, showing how weight updates are represented as products of low-rank matrices A and B, significantly reducing trainable parameters while preserving model expressiveness.*

### 1.2.5 Policy Context: National Development Strategy and AI Strategy

This research is situated within Zimbabwe's broader policy framework for economic development and digital transformation. The National Development Strategy (NDS1 2021-2025, NDS2 2026-2030) identifies financial inclusion as a critical enabler of inclusive economic growth, with specific targets to increase formal credit access to over 90% of adults by 2030 (Government of Zimbabwe, 2021).

The newly launched National AI Strategy (2026-2030) provides additional impetus for AI-driven innovation in financial services. The strategy emphasises responsible AI deployment, ethical considerations, and the development of locally relevant AI solutions that address Zimbabwe's unique challenges. Financial inclusion is identified as a priority application area, with specific focus on leveraging AI to expand credit access to underserved populations (Ministry of ICT, Postal and Courier Services, 2026).

Furthermore, the Reserve Bank of Zimbabwe's National Payment Systems Strategy (2020-2024) and its successor framework prioritise digital financial services innovation, data-driven risk assessment, and regulatory frameworks that balance innovation with consumer protection (Reserve Bank of Zimbabwe, 2020).

## 1.3 Problem Statement

Despite significant advances in financial inclusion in Zimbabwe, with 84% of adults now having access to formal financial services, 89% of the population remains excluded from formal credit facilities. Traditional credit scoring systems, which rely on conventional financial histories, systematically exclude the unbanked and underbanked populations who lack formal banking relationships. This exclusion perpetuates cycles of poverty, constrains entrepreneurship, and limits economic mobility, particularly among rural populations, women, youth, and informal sector workers.

While alternative data sources—including mobile money transactions, utility payments, and digital commerce activities—offer unprecedented opportunities to assess creditworthiness of previously excluded populations, the computational complexity and resource requirements of sophisticated machine learning models capable of processing such diverse data pose significant deployment challenges in resource-constrained environments.

Existing credit scoring approaches in Zimbabwe face three critical limitations:

1. **Data Poverty:** Traditional credit bureaus cover only 15% of adults, leaving the majority of the population without credit histories.

2. **Computational Constraints:** Sophisticated machine learning models require substantial computational resources, making deployment in resource-constrained environments economically unviable.

3. **Fairness and Bias Concerns:** Alternative data credit scoring systems risk perpetuating or amplifying existing socio-economic biases if not carefully designed and evaluated.

This research addresses these challenges by investigating the application of Low-Rank Adaptation (LoRA), a parameter-efficient fine-tuning technique, to develop an alternative data credit scoring system that is simultaneously accurate, computationally efficient, fair, and aligned with Zimbabwe's national development priorities.

## 1.4 Research Aim and Objectives

### 1.4.1 Research Aim

The aim of this research is to develop and evaluate a LoRA-enhanced alternative data credit scoring system that expands financial inclusion in Zimbabwe by accurately assessing creditworthiness of unbanked and underbanked populations while maintaining computational efficiency, fairness, and alignment with national development strategies.

### 1.4.2 Research Objectives

To achieve this aim, the following specific objectives are pursued:

**RO1:** To identify and integrate alternative data sources (mobile money, utility bills) into credit scoring.

**RO2:** To develop a credit scoring model incorporating LoRA for efficient parameter customisation.

**RO3:** To compare the performance of the LoRA-based model with traditional and baseline ML models.

**RO4:** To examine the potential contribution of the LoRA framework to financial inclusion under the NDS.

## 1.5 Research Questions

This research addresses the following research questions:

**RQ1:** How can alternative data sources available in Zimbabwe's digital financial ecosystem be effectively synthesised to construct comprehensive creditworthiness profiles for unbanked and underbanked populations?

**RQ2:** To what extent does LoRA-enhanced fine-tuning maintain predictive performance while reducing computational requirements compared to full fine-tuning approaches in alternative data credit scoring?

**RQ3:** What fairness and bias characteristics emerge in LoRA-enhanced alternative data credit scoring systems, and how can these be mitigated to ensure equitable credit access across demographic groups?

**RQ4:** How can LoRA-enhanced alternative data credit scoring systems be designed and deployed in alignment with Zimbabwe's national development priorities, regulatory frameworks, and ethical AI principles?

## 1.6 Significance of the Study

This research makes significant contributions across theoretical, methodological, practical, and policy dimensions:

### 1.6.1 Theoretical Contributions

**Advancement of Parameter-Efficient Fine-Tuning Theory:** This study extends the theoretical understanding of LoRA's applicability beyond natural language processing to structured financial data and time-series analysis, contributing to the broader PEFT literature.

**Alternative Data Credit Scoring Framework:** The research develops a comprehensive theoretical framework for alternative data credit scoring in developing economy contexts, addressing data poverty, computational constraints, and fairness considerations.

### 1.6.2 Methodological Contributions

**Synthetic Data Generation Methodology:** The study develops a novel synthetic data generation framework that realistically simulates Zimbabwe's digital financial ecosystem, providing a replicable methodology for alternative data research in data-scarce environments.

**Multi-Dimensional Evaluation Framework:** The research proposes a comprehensive evaluation framework that simultaneously assesses predictive performance, computational efficiency, fairness, and policy alignment—addressing the multi-faceted requirements of responsible AI deployment in financial services.

### 1.6.3 Practical Contributions

**Financial Inclusion Impact:** By demonstrating the feasibility of accurate, efficient, and fair alternative data credit scoring, this research directly contributes to expanding credit access to Zimbabwe's unbanked and underbanked populations, potentially impacting millions of individuals and addressing the USD 1.5 billion MSME financing gap.

**Deployment Viability:** The computational efficiency achieved through LoRA makes sophisticated AI-driven credit assessment economically viable for deployment by Zimbabwean financial institutions, microfinance organisations, and fintech companies operating in resource-constrained environments.

**Responsible AI Implementation:** The research provides practical guidance for implementing AI-driven credit scoring systems that balance innovation with ethical considerations, consumer protection, and regulatory compliance.

### 1.6.4 Policy Contributions

**National Development Strategy Alignment:** This study directly supports Zimbabwe's NDS1 and NDS2 objectives for financial inclusion, demonstrating how AI technologies can be leveraged to achieve national development targets.

**AI Strategy Implementation:** The research provides a concrete case study for implementing Zimbabwe's National AI Strategy in the financial services sector, offering insights for policy-makers, regulators, and industry stakeholders.

**Regulatory Framework Development:** The study's findings inform the development of regulatory frameworks for alternative data credit scoring, addressing data privacy, algorithmic fairness, and consumer protection considerations.

## 1.7 Scope and Limitations

### 1.7.1 Scope

This research focuses on:

- **Geographic Scope:** Zimbabwe's financial ecosystem, with particular attention to mobile money, utility payments, and digital commerce data sources prevalent in the country.

- **Technical Scope:** LoRA-enhanced transformer-based models for credit scoring, with comparative analysis against baseline machine learning approaches.

- **Data Scope:** Synthetic alternative data encompassing mobile money transactions, utility payments, digital commerce activities, and demographic information.

- **Evaluation Scope:** Multi-dimensional evaluation covering predictive performance (AUC-ROC, accuracy, precision, recall), computational efficiency (trainable parameters, training time, memory, inference latency), fairness metrics (demographic parity, equalized odds, disparate impact), and policy alignment.

### 1.7.2 Limitations

**Synthetic Data Limitations:** Due to data privacy regulations and limited access to real-world alternative data, this study relies on synthetic data generation. While the synthetic data is designed to realistically simulate Zimbabwe's financial ecosystem based on available statistics and domain expertise, it may not fully capture all complexities of real-world data.

**Temporal Scope:** The research focuses on a specific time period (2024-2026) and may not fully account for rapid changes in Zimbabwe's financial landscape, regulatory environment, or technological infrastructure.

**Deployment Validation:** While the research demonstrates technical feasibility and computational efficiency, full-scale deployment validation with real users and real-world credit outcomes is beyond the scope of this study and represents an important direction for future research.

**Generalisability:** Findings are primarily applicable to Zimbabwe's context and similar developing economies with comparable financial ecosystems. Generalisability to other contexts requires careful consideration of local conditions.

**Model Architecture:** The research focuses specifically on LoRA as a PEFT technique. While comparative analysis with other PEFT methods is included, comprehensive exploration of all possible model architectures and fine-tuning approaches is beyond the scope.

## 1.8 Dissertation Structure

This dissertation is organised into seven chapters:

**Chapter 1: Introduction and Background to the Study** provides the research context, problem statement, objectives, research questions, significance, and scope.

**Chapter 2: Literature Review** presents a comprehensive review of theoretical frameworks, traditional and alternative credit scoring approaches, machine learning techniques, parameter-efficient fine-tuning, financial inclusion in developing economies, and ethical considerations.

**Chapter 3: Research Methodology** details the research philosophy, design, data collection and preparation, model development, and evaluation framework.

**Chapter 4: Data Generation and Preparation** describes the synthetic data generation process, data characteristics, preprocessing steps, and quality assurance measures.

**Chapter 5: Model Development and Implementation** presents the LoRA-enhanced credit scoring model architecture, training procedures, hyperparameter optimisation, and implementation details.

**Chapter 6: Results and Discussion** reports the findings across predictive performance, computational efficiency, fairness evaluation, and policy alignment dimensions, with critical discussion and interpretation.

**Chapter 7: Conclusion and Recommendations** summarises key findings, discusses implications, acknowledges limitations, and provides recommendations for future research and policy development.

---

# CHAPTER 2: LITERATURE REVIEW

## 2.1 Introduction

This chapter presents a comprehensive review of the theoretical and empirical literature underpinning this research. The review is organised into eight thematic sections that collectively establish the conceptual foundation for LoRA-enhanced alternative data credit scoring in Zimbabwe's context.

The chapter begins by establishing the theoretical framework, drawing on financial inclusion theory, information asymmetry theory, and responsible AI principles. It then examines traditional credit scoring systems and their limitations, followed by an exploration of alternative data sources and their application in credit assessment. The review subsequently addresses machine learning approaches in credit scoring, with particular focus on parameter-efficient fine-tuning and LoRA. Financial inclusion challenges in developing economies, with specific attention to Zimbabwe's context, are then examined. The chapter concludes by addressing ethical and regulatory considerations and identifying research gaps that this study aims to address.

## 2.2 Theoretical Framework

### 2.2.1 Financial Inclusion Theory

Financial inclusion, defined as access to and usage of affordable, appropriate, and accessible financial services, has emerged as a critical development priority globally (Demirgüç-Kunt et al., 2018). Theoretical frameworks for financial inclusion emphasise three dimensions: access (availability of financial services), usage (actual utilisation of services), and quality (relevance and appropriateness of services to users' needs).

The capability approach, developed by Sen (1999), provides a foundational theoretical lens for understanding financial inclusion. This perspective emphasises that financial inclusion should be evaluated not merely by access to services, but by the extent to which such access enhances individuals' capabilities to achieve valued functionings—including economic participation, asset accumulation, risk management, and poverty reduction.

In developing economy contexts, financial inclusion theory recognises structural barriers including geographic remoteness, lack of formal identification, limited financial literacy, and absence of collateral or credit histories (Demirgüç-Kunt and Klapper, 2013). These barriers create a "missing middle" where individuals have basic financial access (e.g., mobile money accounts) but remain excluded from credit facilities that could enable economic mobility.

### 2.2.2 Information Asymmetry and Credit Markets

Credit markets are characterised by fundamental information asymmetries between lenders and borrowers. Stiglitz and Weiss (1981) demonstrated that these asymmetries lead to adverse selection and moral hazard problems, resulting in credit rationing where some borrowers are denied credit even when willing to pay higher interest rates.

Traditional credit scoring systems address information asymmetry by synthesising observable signals of creditworthiness—primarily past credit behaviour—into standardised risk assessments. However, in contexts where large populations lack credit histories, information asymmetry is exacerbated, leading to systematic exclusion of potentially creditworthy borrowers (Karlan and Zinman, 2009).

Alternative data represents a potential solution to this information asymmetry problem by providing new signals of creditworthiness. Theoretical work by Berg et al. (2020) demonstrates that digital footprint data can reduce information asymmetry and improve credit allocation efficiency, particularly for borrowers without traditional credit histories.

### 2.2.3 Responsible AI and Algorithmic Fairness

The deployment of AI systems in high-stakes domains such as credit scoring raises critical ethical considerations. Responsible AI frameworks emphasise principles including fairness, transparency, accountability, privacy, and human oversight (Jobin et al., 2019).

Algorithmic fairness theory distinguishes between multiple fairness definitions, including demographic parity (equal acceptance rates across groups), equalized odds (equal true positive and false positive rates across groups), and individual fairness (similar individuals receive similar outcomes) (Barocas and Selbst, 2016). These definitions can be mathematically incompatible, requiring context-specific trade-offs.

In credit scoring contexts, fairness considerations are particularly acute given historical patterns of discrimination and the potential for algorithmic systems to perpetuate or amplify existing biases (Fuster et al., 2022). Theoretical frameworks for fair credit scoring emphasise the need to balance predictive accuracy with equitable outcomes across demographic groups, while respecting regulatory requirements such as equal credit opportunity principles.

**[Figure 2.1: Theoretical Framework Integration]**  
*Caption: Conceptual diagram illustrating the integration of financial inclusion theory, information asymmetry theory, and responsible AI principles in the context of alternative data credit scoring.*

## 2.3 Traditional Credit Scoring Systems

### 2.3.1 Evolution and Methodology

Credit scoring has evolved significantly since the introduction of the first credit bureau in the United States in 1899. Modern credit scoring systems, exemplified by the FICO score introduced in 1989, employ statistical models to predict the probability of default based on credit bureau data (Thomas et al., 2002).

Traditional credit scores synthesise multiple dimensions of credit behaviour: payment history (timeliness of past payments), credit utilisation (proportion of available credit used), length of credit history, new credit inquiries, and credit mix (diversity of credit types). These factors are weighted and combined into a single numerical score, typically ranging from 300 to 850, with higher scores indicating lower default risk (Fair Isaac Corporation, 2020).

The statistical foundation of traditional credit scoring typically employs logistic regression, which models the log-odds of default as a linear function of predictor variables. This approach offers interpretability and regulatory compliance advantages, as the contribution of each factor to the final score can be clearly explained (Siddiqi, 2017).

### 2.3.2 Limitations in Developing Economy Contexts

Traditional credit scoring systems encounter fundamental limitations when applied in developing economy contexts. First, credit bureau coverage is typically limited. In sub-Saharan Africa, only 7% of adults have credit bureau records, compared to 67% in high-income economies (World Bank, 2020). This "data poverty" systematically excludes the majority of the population from credit access.

Second, traditional credit scoring assumes stable formal employment and banking relationships, which are atypical in economies characterised by high informality. In Zimbabwe, where informal economic activities account for approximately 60% of GDP, conventional employment and banking data fail to capture the economic realities of most individuals (Zimbabwe National Statistics Agency, 2023).

Third, traditional credit scoring systems are vulnerable to "thin file" and "no file" problems. Individuals with limited or no credit histories receive low scores or cannot be scored at all, regardless of their actual creditworthiness. This creates a paradox: those most in need of credit to escape poverty are systematically excluded due to lack of credit histories (Hurley and Adebayo, 2016).

**[Table 2.1: Limitations of Traditional Credit Scoring in Zimbabwe]**

| **Limitation** | **Description** | **Impact on Financial Inclusion** |
|----------------|-----------------|-----------------------------------|
| Limited Credit Bureau Coverage | Only 15% of adults have credit bureau records | 85% of population cannot be assessed using traditional methods |
| Informal Economy Dominance | 60% of GDP from informal activities | Conventional employment/banking data unavailable for majority |
| Thin File/No File Problem | Insufficient credit history for scoring | Creditworthy individuals excluded due to lack of data |
| Geographic Bias | Credit bureaus concentrated in urban areas | Rural populations disproportionately excluded |
| Static Assessment | Scores based on historical data only | Fails to capture current financial behaviours |

*Caption: Summary of key limitations of traditional credit scoring systems in Zimbabwe's context and their implications for financial inclusion.*

## 2.4 Alternative Data in Credit Assessment

### 2.4.1 Definition and Taxonomy

Alternative data encompasses non-traditional information sources that can provide insights into creditworthiness. Berg et al. (2020) provide a comprehensive taxonomy of alternative data sources:

**Digital Footprint Data:** Mobile phone usage patterns, telecommunications payment histories, internet browsing behaviours, and social media activities.

**Transactional Data:** Mobile money transactions, digital payment patterns, e-commerce activities, and peer-to-peer transfer behaviours.

**Utility and Bill Payment Data:** Electricity, water, telecommunications, and rent payment histories.

**Psychometric Data:** Personality assessments, cognitive ability tests, and behavioural surveys.

**Geospatial Data:** Location patterns, mobility behaviours, and geographic context.

In Zimbabwe's context, mobile money transaction data represents the most significant alternative data source. With mobile money penetration exceeding 70% of adults and transaction volumes surpassing USD 40 billion annually, mobile money data provides rich longitudinal records of financial behaviours for previously unbanked populations (Reserve Bank of Zimbabwe, 2024).

### 2.4.2 Predictive Power of Alternative Data

Empirical research demonstrates that alternative data can significantly enhance credit risk prediction, particularly for thin-file and no-file populations. Berg et al. (2020), analysing data from a German e-commerce platform, found that digital footprint data improved default prediction accuracy by 30% compared to traditional credit bureau scores alone.

Björkegren and Grissen (2020) examined mobile phone data in a developing economy context, demonstrating that call detail records could predict loan defaults with accuracy comparable to credit bureau scores. Importantly, the predictive power of mobile phone data was strongest for individuals without credit bureau records, suggesting complementarity between traditional and alternative data sources.

In the African context, Blumenstock et al. (2015) demonstrated that mobile phone metadata could predict wealth and consumption patterns with high accuracy, suggesting potential applications in credit assessment. Similarly, research by FSD Africa (2019) across multiple African countries found that mobile money transaction patterns were strongly predictive of creditworthiness, with payment consistency and transaction diversity emerging as particularly important signals.

**[Figure 2.2: Predictive Power of Alternative Data Sources]**  
*Caption: Comparative analysis of predictive accuracy (AUC-ROC) for different data sources in credit scoring, showing the complementary value of alternative data, particularly for thin-file populations. Based on synthesis of Berg et al. (2020) and Björkegren and Grissen (2020).*

### 2.4.3 Alternative Data in Zimbabwe

Zimbabwe's digital financial ecosystem provides several alternative data sources relevant to credit assessment:

**Mobile Money Data:** EcoCash, the dominant mobile money platform, processes over 90% of mobile money transactions in Zimbabwe. Transaction patterns including frequency, volume, consistency, peer-to-peer transfers, merchant payments, and savings behaviours provide rich signals of financial discipline (Reserve Bank of Zimbabwe, 2024).

**Utility Payment Data:** ZESA (electricity utility) and municipal water services maintain payment records that demonstrate payment consistency and financial management capabilities. Telecommunications companies similarly track airtime purchase patterns and bill payment behaviours.

**Digital Commerce Data:** Growing e-commerce platforms and digital marketplaces generate transaction data that can inform creditworthiness assessments, particularly for entrepreneurs and small business owners.

**Remittance Patterns:** Zimbabwe receives substantial remittances from the diaspora, with formal remittance channels generating data on receipt frequency and amounts that may signal financial stability.

However, access to these alternative data sources faces challenges including data privacy regulations, commercial sensitivities, lack of data-sharing infrastructure, and absence of standardised data formats. These challenges motivate the use of synthetic data in this research, designed to realistically simulate Zimbabwe's alternative data ecosystem.

## 2.5 Machine Learning in Credit Scoring

### 2.5.1 Evolution of ML Approaches

Machine learning applications in credit scoring have evolved through several generations. Early approaches employed logistic regression and linear discriminant analysis, offering interpretability but limited capacity to capture non-linear relationships (Hand and Henley, 1997).

The second generation introduced decision trees and ensemble methods including random forests and gradient boosting machines. These approaches demonstrated superior predictive performance by capturing non-linear interactions and complex patterns in credit data (Khandani et al., 2010). XGBoost, in particular, has become widely adopted in credit scoring due to its strong performance and relative interpretability through feature importance measures (Chen and Guestrin, 2016).

The third generation leverages deep learning architectures including neural networks, recurrent neural networks (RNNs), and long short-term memory (LSTM) networks. These approaches excel at processing sequential data and capturing temporal patterns in transaction histories (Sirignano et al., 2016). However, deep learning models face challenges including high computational requirements, limited interpretability, and vulnerability to overfitting in small-sample contexts.

### 2.5.2 Transformer Models and Sequential Data

Transformer architectures, introduced by Vaswani et al. (2017), have revolutionised sequential data processing through self-attention mechanisms that capture long-range dependencies. While initially developed for natural language processing, transformers have demonstrated strong performance in financial time-series analysis and transaction sequence modelling (Li et al., 2023).

BERT (Bidirectional Encoder Representations from Transformers), introduced by Devlin et al. (2019), employs masked language modelling to learn contextual representations. Adaptations of BERT for financial data, such as FinBERT, have shown promise in credit risk assessment by capturing complex patterns in transaction sequences (Araci, 2019).

However, transformer models typically contain millions or billions of parameters, requiring substantial computational resources for training and deployment. This computational intensity poses significant barriers for deployment in resource-constrained environments characteristic of developing economies.

### 2.5.3 Challenges in Developing Economy Deployment

Deploying sophisticated machine learning models in developing economy contexts faces several challenges:

**Computational Constraints:** Limited access to high-performance computing infrastructure, including GPUs and cloud computing resources, constrains the feasibility of training and deploying large-scale models.

**Energy Costs:** High energy costs and unreliable electricity supply in many developing economies make energy-intensive model training economically unviable.

**Technical Expertise:** Shortage of machine learning expertise and data science talent limits local capacity for model development and maintenance.

**Data Quality:** Alternative data sources may suffer from inconsistencies, missing values, and lack of standardisation, requiring extensive preprocessing and quality assurance.

These challenges motivate the exploration of parameter-efficient approaches that maintain model performance while dramatically reducing computational requirements.

## 2.6 Parameter-Efficient Fine-Tuning and LoRA

### 2.6.1 The Fine-Tuning Challenge

Transfer learning, where models pre-trained on large datasets are fine-tuned for specific tasks, has become standard practice in machine learning. However, full fine-tuning—updating all model parameters—requires substantial computational resources and storage, as separate copies of the entire model must be maintained for each task (Howard and Ruder, 2018).

For large transformer models, full fine-tuning presents significant challenges. A model with 100 million parameters requires approximately 400MB of storage per task-specific version. Training requires backpropagation through all layers, demanding substantial GPU memory and training time. These requirements are prohibitive for deployment in resource-constrained environments.

### 2.6.2 Parameter-Efficient Fine-Tuning Techniques

Parameter-Efficient Fine-Tuning (PEFT) techniques address these challenges by updating only a small subset of parameters while keeping the majority of the pre-trained model frozen. Several PEFT approaches have been proposed:

**Adapter Layers:** Houlsby et al. (2019) introduced adapter modules—small neural network layers inserted between transformer layers. Only adapter parameters are updated during fine-tuning, reducing trainable parameters by 90-95% while maintaining performance.

**Prefix Tuning:** Li and Liang (2021) proposed learning task-specific prefix vectors that are prepended to the input, modifying model behaviour without updating the base model parameters.

**Prompt Tuning:** Lester et al. (2021) demonstrated that learning soft prompts—continuous task-specific vectors—can achieve performance comparable to full fine-tuning with minimal parameter updates.

**LoRA (Low-Rank Adaptation):** Hu et al. (2021) introduced LoRA, which decomposes weight updates into low-rank matrices, achieving dramatic parameter reduction while maintaining or improving performance.

### 2.6.3 LoRA: Methodology and Advantages

LoRA is based on the hypothesis that weight updates during fine-tuning have low intrinsic dimensionality. Rather than updating the full weight matrix W, LoRA represents the update as a product of two low-rank matrices:

**W' = W + BA**

where W is the original pre-trained weight matrix (frozen), B is a d×r matrix, A is an r×k matrix, and r << min(d,k) is the rank.

This decomposition reduces trainable parameters from d×k to (d+k)×r. For example, with d=1024, k=1024, and r=8, trainable parameters are reduced from 1,048,576 to 16,384—a 98.4% reduction.

**Key advantages of LoRA include:**

**Dramatic Parameter Reduction:** Reducing trainable parameters by 99%+ while maintaining performance makes deployment feasible in resource-constrained environments.

**Storage Efficiency:** Only the small A and B matrices need to be stored per task, enabling efficient multi-task deployment.

**Training Efficiency:** Reduced parameters accelerate training and reduce memory requirements, enabling training on consumer-grade hardware.

**No Inference Latency:** The low-rank update can be merged into the original weights (W' = W + BA) at deployment, adding no inference latency.

**Modular Deployment:** Different LoRA modules can be swapped for different tasks without modifying the base model.

**[Figure 2.3: LoRA Architecture and Parameter Reduction]**  
*Caption: Detailed illustration of LoRA's low-rank decomposition showing how weight updates are represented as products of low-rank matrices A and B, with comparison of parameter counts between full fine-tuning and LoRA approaches.*

### 2.6.4 LoRA Applications and Empirical Results

Hu et al. (2021) demonstrated LoRA's effectiveness across multiple natural language processing tasks, achieving performance comparable to or exceeding full fine-tuning while reducing trainable parameters by 10,000× and GPU memory requirements by 3×.

Subsequent research has extended LoRA to various domains. Lialin et al. (2023) provided theoretical analysis of LoRA's effectiveness, demonstrating that low-rank updates can approximate full-rank updates for many tasks. Zhang et al. (2023) explored optimal rank selection strategies, finding that ranks between 4 and 16 typically provide optimal trade-offs between efficiency and performance.

However, LoRA's application to financial data and credit scoring remains limited. This research addresses this gap by investigating LoRA's effectiveness for alternative data credit scoring in developing economy contexts.

## 2.7 Financial Inclusion in Developing Economies

### 2.7.1 Global Financial Inclusion Landscape

The World Bank's Global Findex Database reveals significant progress in financial inclusion globally, with 76% of adults having accounts at financial institutions or mobile money providers in 2021, up from 51% in 2011 (Demirgüç-Kunt et al., 2022). However, substantial disparities persist across regions and demographic groups.

Sub-Saharan Africa has experienced the most dramatic transformation, driven primarily by mobile money adoption. Mobile money account ownership increased from 12% in 2014 to 33% in 2021, with countries such as Kenya, Tanzania, and Zimbabwe achieving mobile money penetration exceeding 70% of adults (Demirgüç-Kunt et al., 2022).

However, account ownership does not automatically translate to credit access. Globally, only 22% of adults in developing economies have borrowed from formal financial institutions, compared to 45% in high-income economies. This credit access gap is particularly pronounced in sub-Saharan Africa, where only 11% of adults have formal credit access (World Bank, 2020).

### 2.7.2 Zimbabwe's Financial Inclusion Journey

Zimbabwe's financial inclusion trajectory reflects both remarkable progress and persistent challenges. The FinScope Consumer Survey 2024 reveals that 84% of adults have access to formal financial services, representing substantial growth from 69% in 2014 (FinMark Trust, 2024).

This progress has been driven primarily by mobile money adoption, led by EcoCash, which launched in 2011 and rapidly achieved market dominance. By 2024, over 70% of adults had mobile money accounts, with transaction volumes exceeding USD 40 billion annually—approximately 150% of GDP (Reserve Bank of Zimbabwe, 2024).

However, credit access remains severely constrained. Only 11% of adults have access to formal credit facilities, with the majority relying on informal sources including family, friends, and informal moneylenders. The MSME sector, which employs over 5 million people, faces a financing gap of approximately USD 1.5 billion, constraining business growth and job creation (World Bank, 2023).

**[Table 2.2: Zimbabwe Financial Inclusion Indicators (2014-2024)]**

| **Indicator** | **2014** | **2019** | **2024** | **Target 2030** |
|---------------|----------|----------|----------|-----------------|
| Overall Financial Inclusion | 69% | 77% | 84% | 95% |
| Mobile Money Account Ownership | 23% | 58% | 71% | 85% |
| Bank Account Ownership | 30% | 35% | 38% | 60% |
| Formal Credit Access | 8% | 9% | 11% | 90% |
| MSME Credit Access | 12% | 15% | 18% | 70% |
| Rural Financial Inclusion | 52% | 65% | 72% | 90% |

*Caption: Evolution of financial inclusion indicators in Zimbabwe showing progress in overall inclusion and mobile money adoption, but persistent gaps in credit access. Data sources: FinScope Consumer Surveys 2014-2024, Reserve Bank of Zimbabwe.*

### 2.7.3 Barriers to Credit Access

Research identifies multiple barriers to credit access in Zimbabwe and similar developing economies:

**Lack of Credit Histories:** The majority of the population lacks formal credit histories, making traditional credit assessment impossible (Reserve Bank of Zimbabwe, 2024).

**Collateral Requirements:** Traditional lending requires collateral, which many potential borrowers lack, particularly in contexts of insecure property rights and informal land tenure.

**High Transaction Costs:** Small loan sizes characteristic of MSME and individual lending result in high transaction costs relative to loan values, making such lending unprofitable for traditional banks.

**Information Asymmetry:** Lenders lack information about borrowers' creditworthiness, leading to adverse selection and credit rationing (Karlan and Zinman, 2009).

**Geographic Barriers:** Rural populations face limited access to bank branches and financial services infrastructure.

**Gender Disparities:** Women face disproportionate barriers to credit access due to lower financial literacy, limited collateral ownership, and discriminatory social norms (Demirgüç-Kunt et al., 2022).

Alternative data credit scoring represents a potential solution to several of these barriers by providing new information sources that reduce information asymmetry and enable creditworthiness assessment for populations without traditional credit histories.

## 2.8 Ethical and Regulatory Considerations

### 2.8.1 Fairness and Bias in Credit Scoring

Algorithmic credit scoring systems raise significant fairness concerns. Historical patterns of discrimination in credit markets, combined with the potential for algorithmic systems to perpetuate or amplify existing biases, necessitate careful attention to fairness considerations (Fuster et al., 2022).

Bias can enter credit scoring systems through multiple pathways. Training data may reflect historical discrimination, leading to biased predictions. Feature selection may inadvertently incorporate proxies for protected characteristics such as race or gender. Model optimisation focused solely on predictive accuracy may produce disparate impacts across demographic groups (Barocas and Selbst, 2016).

Research demonstrates that alternative data credit scoring can both mitigate and exacerbate fairness concerns. On one hand, alternative data can reduce bias by providing information about creditworthiness for populations systematically excluded by traditional systems. On the other hand, alternative data sources such as social media or geospatial data may encode societal biases, leading to discriminatory outcomes (Hurley and Adebayo, 2016).

### 2.8.2 Fairness Metrics and Trade-offs

Multiple fairness definitions exist, often with mathematical incompatibilities. Key fairness metrics include:

**Demographic Parity:** Equal acceptance rates across demographic groups. Formally, P(Ŷ=1|A=0) = P(Ŷ=1|A=1), where Ŷ is the prediction and A is the protected attribute.

**Equalized Odds:** Equal true positive rates and false positive rates across groups. Formally, P(Ŷ=1|Y=y,A=0) = P(Ŷ=1|Y=y,A=1) for y ∈ {0,1}.

**Disparate Impact Ratio:** Ratio of acceptance rates between groups, with values below 0.8 typically considered discriminatory under US equal credit opportunity regulations.

**Individual Fairness:** Similar individuals receive similar outcomes, regardless of group membership (Dwork et al., 2012).

These definitions can conflict. For example, if base rates of creditworthiness differ across groups, demographic parity and equalized odds cannot both be satisfied (Chouldechova, 2017). This necessitates context-specific trade-offs informed by ethical considerations and regulatory requirements.

### 2.8.3 Privacy and Data Protection

Alternative data credit scoring raises significant privacy concerns. Mobile money transactions, utility payments, and digital commerce activities reveal intimate details of individuals' lives, including spending patterns, social relationships, and geographic movements. The collection, storage, and use of such data must respect privacy rights and comply with data protection regulations (Hurley and Adebayo, 2016).

Zimbabwe's data protection framework is evolving. The Cyber Security and Data Protection Act (2021) establishes principles for data collection, processing, and storage, including requirements for consent, purpose limitation, data minimisation, and security safeguards. However, implementation and enforcement remain nascent, creating uncertainty for alternative data credit scoring deployment.

International frameworks including the European Union's General Data Protection Regulation (GDPR) provide additional guidance, emphasising principles of transparency, accountability, and individual rights including rights to access, rectification, and erasure of personal data (European Union, 2016).

### 2.8.4 Explainability and Transparency

Explainability—the ability to understand and explain how credit scoring models make decisions—is critical for regulatory compliance, consumer trust, and identification of potential biases. However, sophisticated machine learning models, particularly deep neural networks, are often characterised as "black boxes" with limited interpretability (Rudin, 2019).

Explainable AI (XAI) techniques aim to address this challenge. SHAP (SHapley Additive exPlanations) values provide model-agnostic explanations by quantifying each feature's contribution to individual predictions (Lundberg and Lee, 2017). LIME (Local Interpretable Model-agnostic Explanations) generates local approximations of complex models using interpretable models (Ribeiro et al., 2016).

However, XAI techniques face limitations. Explanations may be unstable, varying significantly with small input changes. Post-hoc explanations may not accurately reflect model decision-making processes. Furthermore, explanations must be meaningful to diverse stakeholders including regulators, lenders, and borrowers, requiring careful communication design (Bhatt et al., 2020).

### 2.8.5 Regulatory Frameworks

Credit scoring is subject to extensive regulation in most jurisdictions, addressing consumer protection, anti-discrimination, and data privacy concerns. Key regulatory principles include:

**Equal Credit Opportunity:** Prohibition of discrimination based on protected characteristics including race, gender, age, and marital status.

**Fair Credit Reporting:** Requirements for accuracy, transparency, and dispute resolution in credit reporting.

**Adverse Action Notices:** Requirements to inform applicants of credit denials and provide reasons for adverse decisions.

**Data Privacy:** Restrictions on data collection, storage, and use, with requirements for consent and security safeguards.

In Zimbabwe, the regulatory framework for alternative data credit scoring is emerging. The Reserve Bank of Zimbabwe's regulatory sandbox, established in 2020, provides a controlled environment for testing innovative financial services, including alternative data credit scoring, while ensuring consumer protection and regulatory compliance (Reserve Bank of Zimbabwe, 2020).

## 2.9 Research Gaps

The literature review reveals several research gaps that this study addresses:

**Gap 1: LoRA Application to Financial Data:** While LoRA has demonstrated effectiveness in natural language processing, its application to structured financial data and time-series analysis remains limited. This research extends LoRA to alternative data credit scoring, investigating its effectiveness in a novel domain.

**Gap 2: Developing Economy Context:** Most credit scoring research focuses on advanced economies with mature financial systems. This research addresses the specific challenges and opportunities of alternative data credit scoring in Zimbabwe's developing economy context, characterised by high informality, limited credit bureau coverage, and mobile money dominance.

**Gap 3: Computational Efficiency in Resource-Constrained Environments:** While computational efficiency is recognised as important, limited research explicitly addresses deployment feasibility in resource-constrained environments. This research demonstrates that sophisticated AI-driven credit assessment can be economically viable in developing economy contexts through parameter-efficient approaches.

**Gap 4: Multi-Dimensional Evaluation:** Existing research typically evaluates credit scoring systems primarily on predictive accuracy. This research employs a comprehensive multi-dimensional evaluation framework encompassing predictive performance, computational efficiency, fairness, and policy alignment.

**Gap 5: Synthetic Data for Alternative Credit Scoring:** Access to real-world alternative data is constrained by privacy regulations and commercial sensitivities. This research develops a novel synthetic data generation methodology that realistically simulates Zimbabwe's digital financial ecosystem, providing a replicable approach for alternative data research in data-scarce environments.

## 2.10 Chapter Summary

This chapter has established the theoretical and empirical foundation for LoRA-enhanced alternative data credit scoring in Zimbabwe's context. The review demonstrates that:

1. Financial inclusion theory, information asymmetry theory, and responsible AI principles provide complementary theoretical lenses for understanding alternative data credit scoring.

2. Traditional credit scoring systems, while effective in advanced economies, encounter fundamental limitations in developing economy contexts characterised by limited credit bureau coverage, high informality, and data poverty.

3. Alternative data sources, particularly mobile money transactions, utility payments, and digital commerce activities, offer unprecedented opportunities to assess creditworthiness of previously excluded populations, with empirical evidence demonstrating significant predictive power.

4. Machine learning approaches have evolved from logistic regression to sophisticated transformer architectures, but computational requirements pose significant deployment barriers in resource-constrained environments.

5. LoRA represents a breakthrough in parameter-efficient fine-tuning, reducing trainable parameters by up to 99.7% while maintaining performance, making sophisticated AI-driven credit assessment economically viable in developing economy contexts.

6. Zimbabwe's financial inclusion journey reflects remarkable progress in mobile money adoption but persistent gaps in credit access, with 89% of the population lacking formal credit facilities despite 84% having financial service access.

7. Ethical and regulatory considerations, including fairness, privacy, explainability, and consumer protection, are critical for responsible deployment of alternative data credit scoring systems.

The literature review identifies significant research gaps that this study addresses, particularly regarding LoRA's application to financial data, developing economy contexts, computational efficiency, multi-dimensional evaluation, and synthetic data methodologies. The next chapter presents the research methodology designed to address these gaps.

---

# CHAPTER 3: RESEARCH METHODOLOGY

## 3.1 Introduction

This chapter presents the research methodology employed to develop and evaluate the LoRA-enhanced alternative data credit scoring system. The methodology is designed to address the research objectives and questions identified in Chapter 1, while adhering to rigorous scientific standards and ethical principles.

The chapter is organised into six sections. Section 3.2 establishes the research philosophy and design, situating the study within pragmatist epistemology and mixed-methods approaches. Section 3.3 details data collection and preparation, including the synthetic data generation framework. Section 3.4 describes the model development process, including architecture selection, LoRA implementation, and training procedures. Section 3.5 presents the comprehensive evaluation framework encompassing predictive performance, computational efficiency, fairness, and policy alignment dimensions. Section 3.6 addresses ethical considerations including data privacy, algorithmic fairness, and responsible AI principles. The chapter concludes with a summary in Section 3.7.

## 3.2 Research Philosophy and Design

### 3.2.1 Research Philosophy

This research adopts a **pragmatist epistemological stance**, which emphasises practical problem-solving and the use of multiple methods to address research questions (Creswell and Plano Clark, 2017). Pragmatism is particularly appropriate for applied research in software engineering and artificial intelligence, where the primary criterion for evaluating knowledge claims is their practical utility and effectiveness in addressing real-world problems.

The pragmatist approach recognises that different research questions may require different methodological approaches. Quantitative methods are employed to evaluate model performance, computational efficiency, and fairness metrics. Qualitative analysis is used to interpret findings in relation to policy contexts and stakeholder needs. This methodological pluralism enables comprehensive investigation of the multi-faceted research problem.

### 3.2.2 Research Design

The research employs a **design science research methodology**, which is widely used in information systems and software engineering research (Hevner et al., 2004). Design science research involves the creation and evaluation of innovative artefacts designed to solve identified problems. The methodology comprises seven activities:

1. **Problem Identification and Motivation:** Establishing the specific research problem and justifying the value of a solution (addressed in Chapter 1).

2. **Objectives of a Solution:** Defining the objectives for a solution, derived from problem definition and existing knowledge (addressed in Chapter 1).

3. **Design and Development:** Creating the artefact—in this case, the LoRA-enhanced alternative data credit scoring system (addressed in Chapters 4 and 5).

4. **Demonstration:** Demonstrating the use of the artefact to solve instances of the problem (addressed in Chapter 6).

5. **Evaluation:** Observing and measuring how well the artefact supports a solution to the problem (addressed in Chapter 6).

6. **Communication:** Communicating the problem, artefact, evaluation, and contributions to relevant audiences (addressed throughout the dissertation).

The research design comprises three phases:

**Phase 1: Data Generation and Preparation** involves developing a synthetic data generation framework that realistically simulates Zimbabwe's digital financial ecosystem, including mobile money transactions, utility payments, digital commerce activities, and demographic information.

**Phase 2: Model Development** involves implementing the LoRA-enhanced credit scoring model, including base model selection, LoRA configuration, training procedures, and hyperparameter optimisation.

**Phase 3: Evaluation and Analysis** involves comprehensive evaluation across multiple dimensions including predictive performance, computational efficiency, fairness, and policy alignment, with comparative analysis against baseline approaches.

**[Figure 3.1: Research Design Overview]**  
*Caption: Schematic representation of the three-phase research design showing the flow from data generation through model development to evaluation and analysis, with feedback loops for iterative refinement.*

### 3.2.3 Research Approach

The research employs a **mixed-methods approach**, combining quantitative and qualitative methods to provide comprehensive investigation of the research problem (Johnson and Onwuegbuzie, 2004).

**Quantitative methods** are used to:
- Generate synthetic alternative data with realistic statistical properties
- Train and optimise machine learning models
- Measure predictive performance using standard metrics (AUC-ROC, accuracy, precision, recall, F1-score)
- Assess computational efficiency (trainable parameters, training time, memory consumption, inference latency)
- Evaluate fairness using quantitative metrics (demographic parity, equalized odds, disparate impact ratio)

**Qualitative methods** are used to:
- Analyse policy documents and regulatory frameworks to assess alignment with national development strategies
- Interpret quantitative findings in relation to Zimbabwe's socio-economic context
- Identify practical implications and recommendations for stakeholders

This mixed-methods approach enables triangulation, where findings from multiple methods are compared to enhance validity and provide richer insights.

## 3.3 Data Collection and Preparation

### 3.3.1 Rationale for Synthetic Data

This research employs synthetic data rather than real-world data for several reasons:

**Privacy and Regulatory Constraints:** Real-world alternative data, particularly mobile money transactions and utility payments, contains sensitive personal information. Access to such data is restricted by privacy regulations including Zimbabwe's Cyber Security and Data Protection Act (2021) and commercial confidentiality agreements.

**Data Availability:** While mobile money and utility payment data exist, they are not publicly available for research purposes. Negotiating data access agreements with multiple organisations (mobile money providers, utility companies, telecommunications firms) would require extensive time and resources beyond the scope of this research.

**Controlled Experimentation:** Synthetic data enables controlled experimentation with known ground truth, facilitating rigorous evaluation of model performance and fairness characteristics.

**Reproducibility:** Synthetic data generation procedures can be documented and shared, enabling reproducibility and validation by other researchers.

**Ethical Considerations:** Using synthetic data eliminates risks of privacy breaches, re-identification, and potential harm to individuals whose data might otherwise be used.

Synthetic data generation for machine learning research is an established methodology, particularly in contexts where real data is unavailable or sensitive (Jordon et al., 2022). The key challenge is ensuring that synthetic data realistically simulates the statistical properties, patterns, and complexities of real-world data.

### 3.3.2 Synthetic Data Generation Framework

The synthetic data generation framework is designed to realistically simulate Zimbabwe's digital financial ecosystem. The framework comprises four components:

**Component 1: Demographic Profile Generation**

Demographic profiles are generated to reflect Zimbabwe's population distribution across key dimensions:

- **Age:** Distributed according to Zimbabwe's population pyramid, with concentration in working-age groups (18-65 years)
- **Gender:** Balanced distribution with slight female majority (52%) reflecting national demographics
- **Geographic Location:** Urban (38%) vs. rural (62%) distribution reflecting Zimbabwe National Statistics Agency data
- **Income Level:** Distribution reflecting Zimbabwe's income inequality, with Gini coefficient approximation
- **Employment Status:** Formal employment (30%), informal employment (50%), unemployed (20%)

**Component 2: Mobile Money Transaction Generation**

Mobile money transactions are generated using time-series models that capture realistic patterns:

- **Transaction Frequency:** Poisson distribution with mean varying by income level and employment status
- **Transaction Amounts:** Log-normal distribution with parameters varying by transaction type
- **Transaction Types:** Peer-to-peer transfers (60%), merchant payments (25%), bill payments (10%), savings (5%)
- **Temporal Patterns:** Weekly and monthly cycles reflecting salary payment patterns and consumption behaviours
- **Consistency:** Autocorrelation structures capturing payment regularity

**Component 3: Utility Payment Generation**

Utility payment histories are generated for electricity, water, and telecommunications:

- **Payment Frequency:** Monthly billing cycles with realistic variation in payment timing
- **Payment Amounts:** Reflecting typical consumption patterns by household size and income level
- **Payment Consistency:** Probability of on-time payment varying by income level and financial discipline
- **Arrears Patterns:** Realistic accumulation and clearance of arrears

**Component 4: Credit Outcome Generation**

Credit outcomes (default vs. non-default) are generated using a probabilistic model that incorporates:

- **Transaction Consistency:** Higher consistency reduces default probability
- **Payment History:** On-time utility payments reduce default probability
- **Income Stability:** Regular income patterns reduce default probability
- **Financial Stress Indicators:** High transaction volatility increases default probability
- **Demographic Factors:** Controlled relationships to enable fairness evaluation

The default probability is modelled as:

**P(default) = logistic(β₀ + β₁×consistency + β₂×payment_history + β₃×income_stability + β₄×volatility + ε)**

where coefficients are calibrated to produce realistic default rates (approximately 15-20% for the overall population, varying by risk segments).

**[Table 3.1: Synthetic Data Generation Parameters]**

| **Component** | **Variable** | **Distribution/Model** | **Parameters** | **Rationale** |
|---------------|--------------|------------------------|----------------|---------------|
| Demographics | Age | Truncated Normal | μ=35, σ=12, range=[18,65] | Zimbabwe population pyramid |
| Demographics | Gender | Bernoulli | p(female)=0.52 | National gender distribution |
| Demographics | Location | Bernoulli | p(urban)=0.38 | Urban/rural distribution |
| Demographics | Income | Log-normal | μ=5.5, σ=0.8 | Income inequality approximation |
| Mobile Money | Transaction Frequency | Poisson | λ varies by income | Observed mobile money patterns |
| Mobile Money | Transaction Amount | Log-normal | μ, σ vary by type | Realistic amount distributions |
| Mobile Money | Consistency | Beta | α, β vary by profile | Payment regularity patterns |
| Utilities | Payment Timing | Normal | μ=due_date, σ=5 days | Realistic payment variation |
| Utilities | Arrears | Markov Chain | Transition probabilities | Arrears accumulation/clearance |
| Credit Outcome | Default | Logistic Regression | Coefficients calibrated | Realistic default rates |

*Caption: Summary of synthetic data generation parameters and their calibration rationale, ensuring realistic simulation of Zimbabwe's digital financial ecosystem.*

### 3.3.3 Data Quality Assurance

Multiple quality assurance measures are implemented to ensure synthetic data realism:

**Statistical Validation:** Generated data distributions are compared against available statistics from Reserve Bank of Zimbabwe, FinScope surveys, and academic literature to ensure alignment.

**Domain Expert Review:** Synthetic data samples are reviewed by domain experts (financial services professionals, microfinance practitioners) to assess realism and identify potential issues.

**Correlation Analysis:** Relationships between variables (e.g., income and transaction frequency, payment consistency and default rates) are validated against expected patterns.

**Edge Case Testing:** Extreme cases (very high/low income, perfect/poor payment histories) are examined to ensure realistic behaviour.

### 3.3.4 Data Preprocessing

Generated synthetic data undergoes preprocessing to prepare it for model training:

**Feature Engineering:** Raw transaction data is transformed into features including:
- Transaction frequency (daily, weekly, monthly)
- Transaction amount statistics (mean, median, standard deviation, percentiles)
- Payment consistency metrics (on-time payment rate, arrears frequency)
- Temporal patterns (day-of-week effects, monthly cycles)
- Volatility measures (coefficient of variation, entropy)

**Normalisation:** Continuous features are normalised using standardisation (z-score normalisation) to ensure comparable scales.

**Encoding:** Categorical variables (gender, location, employment status) are one-hot encoded.

**Sequence Preparation:** Transaction sequences are formatted for input to transformer models, with padding/truncation to fixed lengths.

**Train-Test Split:** Data is split into training (70%), validation (15%), and test (15%) sets using stratified sampling to ensure balanced default rates across splits.

## 3.4 Model Development

### 3.4.1 Base Model Selection

The base model for LoRA enhancement is selected through comparative evaluation of candidate architectures:

**Candidate 1: Logistic Regression** serves as a baseline, representing traditional credit scoring approaches.

**Candidate 2: XGBoost** represents state-of-the-art gradient boosting, widely used in credit scoring.

**Candidate 3: LSTM** captures temporal patterns in transaction sequences using recurrent neural networks.

**Candidate 4: Transformer (BERT-based)** leverages self-attention mechanisms to capture complex patterns in transaction sequences.

Selection criteria include:
- Predictive performance on validation set
- Suitability for LoRA enhancement (transformer architectures)
- Interpretability and explainability
- Computational requirements

Based on preliminary experiments, a **DistilBERT-based transformer** is selected as the base model. DistilBERT, a distilled version of BERT, offers strong performance with reduced computational requirements compared to full BERT (Sanh et al., 2019). The model is pre-trained on financial transaction sequences and then fine-tuned using LoRA for credit scoring.

### 3.4.2 LoRA Implementation

LoRA is implemented by inserting low-rank adaptation modules into the transformer architecture. Specifically, LoRA modules are applied to the query and value projection matrices in the self-attention layers.

**LoRA Configuration:**

- **Rank (r):** The rank of the low-rank matrices is a critical hyperparameter. Values of r ∈ {4, 8, 16, 32} are evaluated, with r=8 selected based on validation performance and efficiency trade-offs.

- **Alpha (α):** The scaling parameter α controls the magnitude of LoRA updates. α=16 is used, following recommendations from Hu et al. (2021).

- **Target Modules:** LoRA is applied to query and value projection matrices in all transformer layers.

- **Dropout:** Dropout rate of 0.1 is applied to LoRA modules to prevent overfitting.

**Implementation Details:**

The implementation uses the Hugging Face Transformers library and the PEFT (Parameter-Efficient Fine-Tuning) library, which provides efficient LoRA implementations. The base DistilBERT model is loaded with frozen weights, and LoRA modules are initialised and attached to target layers.

**[Figure 3.2: LoRA Implementation in Transformer Architecture]**  
*Caption: Detailed diagram showing LoRA module insertion into transformer self-attention layers, with low-rank matrices A and B applied to query and value projections. Frozen base model parameters are shown in grey, trainable LoRA parameters in colour.*

### 3.4.3 Training Procedures

**Training Configuration:**

- **Optimiser:** AdamW optimiser with weight decay of 0.01
- **Learning Rate:** 3e-4 with linear warmup over 10% of training steps and cosine decay
- **Batch Size:** 32 (limited by GPU memory constraints)
- **Epochs:** Maximum 20 epochs with early stopping based on validation loss
- **Loss Function:** Binary cross-entropy loss for default prediction
- **Regularisation:** Dropout (0.1), weight decay (0.01), and early stopping

**Training Infrastructure:**

Training is conducted on a single NVIDIA GPU (RTX 3090 or equivalent) to simulate resource-constrained deployment environments. Training time, memory consumption, and energy usage are monitored.

**Hyperparameter Optimisation:**

Hyperparameters including LoRA rank, learning rate, and batch size are optimised using Bayesian optimisation with 50 trials. The objective function is validation AUC-ROC, with computational efficiency as a secondary consideration.

### 3.4.4 Baseline Models

For comparative evaluation, several baseline models are trained:

**Baseline 1: Logistic Regression** using engineered features (transaction statistics, payment consistency metrics).

**Baseline 2: XGBoost** using the same engineered features.

**Baseline 3: LSTM** processing transaction sequences directly.

**Baseline 4: Full Fine-Tuning** of the DistilBERT model (all parameters trainable) for comparison with LoRA.

All baseline models are trained using the same train-test splits and evaluation procedures to ensure fair comparison.

## 3.5 Evaluation Framework

### 3.5.1 Predictive Performance Evaluation

Predictive performance is evaluated using standard binary classification metrics:

**AUC-ROC (Area Under Receiver Operating Characteristic Curve):** Measures the model's ability to discriminate between default and non-default cases across all classification thresholds. AUC-ROC is the primary performance metric.

**Accuracy:** Proportion of correct predictions (both true positives and true negatives).

**Precision:** Proportion of predicted defaults that are actual defaults (positive predictive value).

**Recall (Sensitivity):** Proportion of actual defaults correctly identified (true positive rate).

**F1-Score:** Harmonic mean of precision and recall, providing a balanced measure.

**Confusion Matrix:** Detailed breakdown of true positives, true negatives, false positives, and false negatives.

Performance is evaluated on the held-out test set, with 95% confidence intervals computed using bootstrap resampling (1000 iterations).

**[Table 3.2: Predictive Performance Metrics]**

| **Metric** | **Formula** | **Interpretation** | **Target** |
|------------|-------------|-------------------|-----------|
| AUC-ROC | Area under ROC curve | Discrimination ability | > 0.75 |
| Accuracy | (TP + TN) / (TP + TN + FP + FN) | Overall correctness | > 0.80 |
| Precision | TP / (TP + FP) | Positive predictive value | > 0.70 |
| Recall | TP / (TP + FN) | Sensitivity | > 0.70 |
| F1-Score | 2 × (Precision × Recall) / (Precision + Recall) | Balanced measure | > 0.70 |

*Caption: Predictive performance metrics, their formulas, interpretations, and target thresholds for acceptable performance. TP=True Positives, TN=True Negatives, FP=False Positives, FN=False Negatives.*

### 3.5.2 Computational Efficiency Evaluation

Computational efficiency is evaluated across multiple dimensions:

**Trainable Parameters:** Number of parameters updated during fine-tuning, compared between LoRA and full fine-tuning.

**Training Time:** Wall-clock time required for model training, measured in hours.

**Memory Consumption:** Peak GPU memory usage during training, measured in gigabytes.

**Inference Latency:** Time required to generate predictions for a single sample, measured in milliseconds.

**Energy Consumption:** Estimated energy usage during training, calculated based on GPU power consumption and training time.

**Storage Requirements:** Disk space required to store model parameters, compared between LoRA and full fine-tuning.

These metrics demonstrate the feasibility of deploying the LoRA-enhanced model in resource-constrained environments.

**[Table 3.3: Computational Efficiency Metrics]**

| **Metric** | **Unit** | **Measurement Method** | **Comparison** |
|------------|----------|------------------------|----------------|
| Trainable Parameters | Millions | Parameter count | LoRA vs. Full Fine-Tuning |
| Training Time | Hours | Wall-clock time | LoRA vs. Full Fine-Tuning |
| Memory Consumption | GB | Peak GPU memory | LoRA vs. Full Fine-Tuning |
| Inference Latency | Milliseconds | Average per sample | LoRA vs. Baselines |
| Energy Consumption | kWh | GPU power × time | LoRA vs. Full Fine-Tuning |
| Storage Requirements | MB | Model file size | LoRA vs. Full Fine-Tuning |

*Caption: Computational efficiency metrics and measurement methods for evaluating the resource requirements of LoRA-enhanced credit scoring compared to baseline approaches.*

### 3.5.3 Fairness Evaluation

Fairness is evaluated across demographic dimensions including gender, age, geographic location, and income level. Multiple fairness metrics are computed:

**Demographic Parity Difference (DPD):** Difference in acceptance rates between demographic groups.

**DPD = P(Ŷ=1|A=0) - P(Ŷ=1|A=1)**

where Ŷ is the prediction and A is the protected attribute. Values close to 0 indicate demographic parity.

**Equalized Odds Difference (EOD):** Maximum difference in true positive rates and false positive rates between groups.

**EOD = max(|TPR₀ - TPR₁|, |FPR₀ - FPR₁|)**

where TPR is true positive rate and FPR is false positive rate. Values close to 0 indicate equalized odds.

**Disparate Impact Ratio (DIR):** Ratio of acceptance rates between groups.

**DIR = P(Ŷ=1|A=0) / P(Ŷ=1|A=1)**

Values between 0.8 and 1.25 are typically considered acceptable under US equal credit opportunity regulations.

**Calibration by Group:** Comparison of predicted probabilities vs. actual default rates across demographic groups to assess whether the model is equally well-calibrated for all groups.

Fairness metrics are computed for multiple demographic dimensions and reported with confidence intervals. Fairness-performance trade-offs are analysed to identify optimal operating points.

**[Figure 3.3: Fairness Evaluation Framework]**  
*Caption: Conceptual framework for fairness evaluation showing multiple fairness metrics (demographic parity, equalized odds, disparate impact) assessed across demographic dimensions (gender, age, location, income), with analysis of fairness-performance trade-offs.*

### 3.5.4 Policy Alignment Evaluation

Policy alignment is evaluated qualitatively through analysis of how the proposed credit scoring system aligns with:

**National Development Strategy (NDS1, NDS2):** Assessment of contribution to financial inclusion targets, MSME financing, and inclusive economic growth objectives.

**National AI Strategy (2026-2030):** Evaluation of alignment with responsible AI principles, ethical considerations, and priority application areas.

**Regulatory Frameworks:** Analysis of compliance with Reserve Bank of Zimbabwe regulations, Cyber Security and Data Protection Act, and consumer protection requirements.

**International Best Practices:** Comparison with international standards for responsible AI in financial services, including fairness, transparency, and accountability principles.

This qualitative analysis draws on policy document review, regulatory framework analysis, and stakeholder perspective consideration.

## 3.6 Ethical Considerations

### 3.6.1 Data Privacy and Protection

Although this research uses synthetic data, ethical principles for data privacy and protection are rigorously applied:

**Data Minimisation:** Only data necessary for credit assessment is generated and used.

**Purpose Limitation:** Synthetic data is used solely for research purposes and is not shared beyond the research team.

**Security Safeguards:** Data is stored securely with access controls and encryption.

**Anonymisation:** Synthetic data contains no real personal information, eliminating re-identification risks.

These principles align with Zimbabwe's Cyber Security and Data Protection Act (2021) and international best practices including GDPR.

### 3.6.2 Algorithmic Fairness

Fairness is a central ethical consideration in this research. Multiple measures are implemented to promote fairness:

**Fairness-Aware Design:** The synthetic data generation process is designed to enable fairness evaluation, with controlled relationships between demographic attributes and credit outcomes.

**Multi-Metric Evaluation:** Multiple fairness metrics are computed to provide comprehensive fairness assessment.

**Bias Mitigation:** If significant biases are detected, mitigation strategies including re-weighting, threshold adjustment, or fairness constraints are explored.

**Transparency:** Fairness evaluation results are reported transparently, including any trade-offs between fairness and predictive performance.

### 3.6.3 Responsible AI Principles

The research adheres to responsible AI principles throughout:

**Transparency:** Model architecture, training procedures, and evaluation methods are documented in detail to enable reproducibility and scrutiny.

**Accountability:** Clear documentation of design decisions, limitations, and potential risks enables accountability.

**Human Oversight:** The research emphasises that AI-driven credit scoring should augment, not replace, human decision-making, with mechanisms for human review and appeal.

**Beneficence:** The research is motivated by the goal of expanding financial inclusion and improving access to credit for underserved populations.

**Non-Maleficence:** Potential harms including privacy breaches, discrimination, and financial exclusion are carefully considered and mitigated.

### 3.6.4 Research Ethics Approval

This research has received ethics approval from the Harare Institute of Technology Research Ethics Committee. The approval confirms that the research design, methodology, and ethical safeguards meet institutional and national ethical standards for research involving human subjects (even though synthetic data is used, the research has implications for real individuals and communities).

## 3.7 Chapter Summary

This chapter has presented the comprehensive research methodology for developing and evaluating the LoRA-enhanced alternative data credit scoring system. Key methodological elements include:

1. **Research Philosophy and Design:** Pragmatist epistemology and design science research methodology, employing mixed methods to address multi-faceted research questions.

2. **Synthetic Data Generation:** A novel framework for generating realistic synthetic alternative data simulating Zimbabwe's digital financial ecosystem, including mobile money transactions, utility payments, and credit outcomes.

3. **Model Development:** Implementation of LoRA-enhanced transformer-based credit scoring, with careful selection of base models, LoRA configuration, and training procedures optimised for resource-constrained environments.

4. **Comprehensive Evaluation:** Multi-dimensional evaluation framework encompassing predictive performance, computational efficiency, fairness, and policy alignment, enabling holistic assessment of the proposed system.

5. **Ethical Considerations:** Rigorous attention to data privacy, algorithmic fairness, and responsible AI principles throughout the research process.

The methodology is designed to produce rigorous, reproducible, and ethically sound research that addresses the identified research gaps and contributes to both theoretical understanding and practical deployment of alternative data credit scoring in developing economy contexts.

The next chapters will present the implementation of this methodology, reporting data generation and preparation (Chapter 4), model development and implementation (Chapter 5), and results and discussion (Chapter 6).

---

# REFERENCES

Araci, D. (2019). FinBERT: Financial sentiment analysis with pre-trained language models. *arXiv preprint arXiv:1908.10063*.

Barocas, S. and Selbst, A.D. (2016). Big data's disparate impact. *California Law Review*, 104, pp.671-732.

Berg, T., Burg, V., Gombović, A. and Puri, M. (2020). On the rise of FinTechs: Credit scoring using digital footprints. *The Review of Financial Studies*, 33(7), pp.2845-2897.

Bhatt, U., Xiang, A., Sharma, S., Weller, A., Taly, A., Jia, Y., Ghosh, J., Puri, R., Moura, J.M. and Eckersley, P. (2020). Explainable machine learning in deployment. In *Proceedings of the 2020 Conference on Fairness, Accountability, and Transparency* (pp.648-657).

Björkegren, D. and Grissen, D. (2020). Behavior revealed in mobile phone usage predicts credit repayment. *The World Bank Economic Review*, 34(3), pp.618-634.

Blumenstock, J., Cadamuro, G. and On, R. (2015). Predicting poverty and wealth from mobile phone metadata. *Science*, 350(6264), pp.1073-1076.

Chen, T. and Guestrin, C. (2016). XGBoost: A scalable tree boosting system. In *Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining* (pp.785-794).

Chouldechova, A. (2017). Fair prediction with disparate impact: A study of bias in recidivism prediction instruments. *Big Data*, 5(2), pp.153-163.

Creswell, J.W. and Plano Clark, V.L. (2017). *Designing and conducting mixed methods research*. 3rd ed. Thousand Oaks, CA: SAGE Publications.

Demirgüç-Kunt, A. and Klapper, L. (2013). Measuring financial inclusion: Explaining variation in use of financial services across and within countries. *Brookings Papers on Economic Activity*, 2013(1), pp.279-340.

Demirgüç-Kunt, A., Klapper, L., Singer, D. and Ansar, S. (2018). *The Global Findex Database 2017: Measuring financial inclusion and the fintech revolution*. Washington, DC: World Bank.

Demirgüç-Kunt, A., Klapper, L., Singer, D. and Ansar, S. (2022). *The Global Findex Database 2021: Financial inclusion, digital payments, and resilience in the age of COVID-19*. Washington, DC: World Bank.

Devlin, J., Chang, M.W., Lee, K. and Toutanova, K. (2019). BERT: Pre-training of deep bidirectional transformers for language understanding. In *Proceedings of the 2019 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies* (pp.4171-4186).

Dwork, C., Hardt, M., Pitassi, T., Reingold, O. and Zemel, R. (2012). Fairness through awareness. In *Proceedings of the 3rd Innovations in Theoretical Computer Science Conference* (pp.214-226).

European Union (2016). *General Data Protection Regulation (GDPR)*. Official Journal of the European Union, L119.

Fair Isaac Corporation (2020). *Understanding your FICO score*. Available at: https://www.fico.com [Accessed 15 January 2026].

FinMark Trust (2024). *FinScope Consumer Survey Zimbabwe 2024*. Johannesburg: FinMark Trust.

FSD Africa (2019). *Alternative credit scoring in Africa: Opportunities and challenges*. Nairobi: FSD Africa.

Fuster, A., Goldsmith-Pinkham, P., Ramadorai, T. and Walther, A. (2022). Predictably unequal? The effects of machine learning on credit markets. *The Journal of Finance*, 77(1), pp.5-47.

Government of Zimbabwe (2021). *National Development Strategy 1 (NDS1) 2021-2025*. Harare: Government of Zimbabwe.

Hand, D.J. and Henley, W.E. (1997). Statistical classification methods in consumer credit scoring: A review. *Journal of the Royal Statistical Society: Series A (Statistics in Society)*, 160(3), pp.523-541.

Hevner, A.R., March, S.T., Park, J. and Ram, S. (2004). Design science in information systems research. *MIS Quarterly*, 28(1), pp.75-105.

Houlsby, N., Giurgiu, A., Jastrzebski, S., Morrone, B., De Laroussilhe, Q., Gesmundo, A., Attariyan, M. and Gelly, S. (2019). Parameter-efficient transfer learning for NLP. In *Proceedings of the 36th International Conference on Machine Learning* (pp.2790-2799).

Howard, J. and Ruder, S. (2018). Universal language model fine-tuning for text classification. In *Proceedings of the 56th Annual Meeting of the Association for Computational Linguistics* (pp.328-339).

Hu, E.J., Shen, Y., Wallis, P., Allen-Zhu, Z., Li, Y., Wang, S., Wang, L. and Chen, W. (2021). LoRA: Low-rank adaptation of large language models. *arXiv preprint arXiv:2106.09685*.

Hurley, M. and Adebayo, J. (2016). *Credit scoring in the era of big data*. Yale Journal of Law and Technology, 18, pp.148-216.

Jobin, A., Ienca, M. and Vayena, E. (2019). The global landscape of AI ethics guidelines. *Nature Machine Intelligence*, 1(9), pp.389-399.

Johnson, R.B. and Onwuegbuzie, A.J. (2004). Mixed methods research: A research paradigm whose time has come. *Educational Researcher*, 33(7), pp.14-26.

Jordon, J., Szpruch, L., Houssiau, F., Bottarelli, M., Cherubin, G., Maple, C., Cohen, S.N. and Weller, A. (2022). Synthetic data: What, why and how? *arXiv preprint arXiv:2205.03257*.

Karlan, D. and Zinman, J. (2009). Observing unobservables: Identifying information asymmetries with a consumer credit field experiment. *Econometrica*, 77(6), pp.1993-2008.

Khandani, A.E., Kim, A.J. and Lo, A.W. (2010). Consumer credit-risk models via machine-learning algorithms. *Journal of Banking & Finance*, 34(11), pp.2767-2787.

Lester, B., Al-Rfou, R. and Constant, N. (2021). The power of scale for parameter-efficient prompt tuning. In *Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing* (pp.3045-3059).

Li, X. and Liang, P. (2021). Prefix-tuning: Optimizing continuous prompts for generation. In *Proceedings of the 59th Annual Meeting of the Association for Computational Linguistics* (pp.4582-4597).

Li, Y., Wang, S., Ding, H., Chen, H., Yang, Y., Yang, Y., Zhao, J., Qin, C., Xiao, J., Xu, Q. and Zheng, B. (2023). FinGPT: Open-source financial large language models. *arXiv preprint arXiv:2306.06031*.

Lialin, V., Deshpande, V. and Rumshisky, A. (2023). Scaling down to scale up: A guide to parameter-efficient fine-tuning. *arXiv preprint arXiv:2303.15647*.

Lundberg, S.M. and Lee, S.I. (2017). A unified approach to interpreting model predictions. In *Advances in Neural Information Processing Systems* (pp.4765-4774).

Ministry of Finance and Economic Development (2025). *2025 National Budget Statement*. Harare: Government of Zimbabwe.

Ministry of ICT, Postal and Courier Services (2026). *National Artificial Intelligence Strategy 2026-2030*. Harare: Government of Zimbabwe.

Reserve Bank of Zimbabwe (2020). *National Payment Systems Strategy 2020-2024*. Harare: Reserve Bank of Zimbabwe.

Reserve Bank of Zimbabwe (2024). *Monetary Policy Statement: January 2024*. Harare: Reserve Bank of Zimbabwe.

Ribeiro, M.T., Singh, S. and Guestrin, C. (2016). "Why should I trust you?" Explaining the predictions of any classifier. In *Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining* (pp.1135-1144).

Rudin, C. (2019). Stop explaining black box machine learning models for high stakes decisions and use interpretable models instead. *Nature Machine Intelligence*, 1(5), pp.206-215.

Sanh, V., Debut, L., Chaumond, J. and Wolf, T. (2019). DistilBERT, a distilled version of BERT: Smaller, faster, cheaper and lighter. *arXiv preprint arXiv:1910.01108*.

Sen, A. (1999). *Development as freedom*. Oxford: Oxford University Press.

Siddiqi, N. (2017). *Intelligent credit scoring: Building and implementing better credit risk scorecards*. 2nd ed. Hoboken, NJ: John Wiley & Sons.

Sirignano, J., Sadhwani, A. and Giesecke, K. (2016). Deep learning for mortgage risk. *arXiv preprint arXiv:1607.02470*.

Stiglitz, J.E. and Weiss, A. (1981). Credit rationing in markets with imperfect information. *American Economic Review*, 71(3), pp.393-410.

Thomas, L.C., Edelman, D.B. and Crook, J.N. (2002). *Credit scoring and its applications*. Philadelphia, PA: SIAM.

Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A.N., Kaiser, Ł. and Polosukhin, I. (2017). Attention is all you need. In *Advances in Neural Information Processing Systems* (pp.5998-6008).

World Bank (2020). *Doing Business 2020: Comparing business regulation in 190 economies*. Washington, DC: World Bank.

World Bank (2023). *Zimbabwe Economic Update: Navigating the storm*. Washington, DC: World Bank.

Zhang, Q., Chen, M., Bukharin, A., He, P., Cheng, Y., Chen, W. and Zhao, T. (2023). AdaLoRA: Adaptive budget allocation for parameter-efficient fine-tuning. *arXiv preprint arXiv:2303.10512*.

Zimbabwe National Statistics Agency (2023). *Labour Force and Child Labour Survey 2022*. Harare: Zimbabwe National Statistics Agency.

---

**END OF CHAPTERS 1-3**
