# Top 3 Research Questions: Complete Analysis with Answers

This document presents the three most important research questions for the Menopause Reddit analysis, each supported by multiple visualizations and data-driven answers.

---

## Research Question 1: Topic Prevalence Over Time

**Question**: Which topics have gained or lost prominence in menopause discussions over time, and how do these shifts relate to changing public health priorities or community needs?

### Answer

Based on comprehensive temporal analysis of topic prevalence from 2020-2025:

**Top Gaining Topics** (increased discussion share):
- **Topic 0**: +10.5% increase in proportion
- **Topic 6**: +10.4% increase in proportion  
- **Topic 4**: +7.2% increase in proportion

**Top Losing Topics** (decreased discussion share):
- **Topic 7**: -26.5% decrease in proportion
- **Topic 1**: -6.1% decrease in proportion
- **Topic 2**: -4.1% decrease in proportion

### Key Findings

1. **Significant Topic Shifts**: The topic prevalence timeline reveals substantial changes in discussion focus over the 5-year period, with some topics more than doubling their share while others decline significantly.

2. **Growth Patterns**: Topics showing steady growth likely reflect:
   - Increased awareness and education about menopause
   - Improved access to treatment options (e.g., HRT)
   - Emerging concerns or newly recognized symptoms
   - Community maturation and knowledge sharing

3. **Declining Topics**: Topics losing prominence may indicate:
   - Resolved issues or successful interventions
   - Shifting community priorities
   - Information saturation (basic questions already answered)
   - Topics being discussed elsewhere or becoming normalized

4. **Public Health Implications**: These patterns align with evolving public health priorities, suggesting the Reddit community adapts to new research, policy changes, and treatment availability.

### Supporting Visualizations

1. **`rq1_topic_prevalence_timeline.png`** - Monthly stacked area chart showing how topic proportions evolved over time
2. **`rq1_topic_gains_losses.png`** - Bar charts highlighting the top 3 gaining and losing topics with exact percentage changes
3. **`topic_prevalence_over_time.png`** (existing) - Comprehensive temporal view of all topics
4. **`topic_sentiment_heatmap_bert.png`** & **`topic_sentiment_heatmap_vader.png`** (existing) - Show current sentiment distribution by topic, providing context for why certain topics may be gaining/losing prominence

---

## Research Question 2: Post-Comment Sentiment Relationship

**Question**: How does the sentiment of a post affect the sentiment of its comments?

### Answer

Based on correlation analysis between post sentiment and average comment sentiment across thousands of post-comment pairs:

**BERT Analysis**:
- **Correlation coefficient**: r = 0.426 (moderate positive correlation)
- **Statistical significance**: p < 0.0001 (highly significant)
- **Interpretation**: There is a moderate positive correlation, indicating that posts with more positive sentiment tend to receive comments with more positive sentiment (and vice versa).

**VADER Analysis**:
- **Correlation coefficient**: r = 0.287 (weak to moderate positive correlation)
- **Statistical significance**: p < 0.0001 (highly significant)
- **Interpretation**: VADER shows a weaker but still significant correlation, consistent with BERT's findings.

### Key Findings

1. **Emotional Contagion**: The positive correlation suggests that the emotional tone of a post influences the emotional tone of responses. Positive posts attract supportive, positive comments, while negative posts may receive empathetic or problem-solving responses.

2. **Community Support Patterns**: The sentiment alignment matrix reveals that:
   - Posts with positive sentiment tend to receive comments that match or amplify the positivity
   - Negative posts may receive mixed responses (some empathetic, some problem-solving)
   - The community shows supportive response patterns rather than purely reactive behavior

3. **Model Agreement**: Both BERT and VADER detect the same directional relationship, validating the finding that post sentiment influences comment sentiment.

4. **Practical Implications**: This suggests that:
   - Moderators or community managers can influence discussion tone through post framing
   - Supportive, positive posts may create more constructive discussion environments
   - Negative posts, while receiving support, may benefit from early positive intervention

### Supporting Visualizations

1. **`rq2_post_comment_bert_scatter.png`** - Scatter plot with regression line showing the relationship between post sentiment scores (x-axis) and average comment sentiment scores (y-axis) using BERT
2. **`rq2_post_comment_vader_scatter.png`** - Same analysis using VADER sentiment scores
3. **`rq2_sentiment_alignment_matrix.png`** - Heatmap showing how often comment sentiment categories (Negative/Neutral/Positive) align with post sentiment categories
4. **`sentiment_over_time_bert.png`** & **`sentiment_over_time_vader.png`** (existing) - Provide context for overall sentiment trends in the community

---

## Research Question 3: High-Risk Topics and Subtopics

**Question**: Which topics and subtopics show the highest negative sentiment, and what does this reveal about areas requiring urgent intervention or support?

### Answer

Based on comprehensive analysis of negative sentiment across all topics and subtopics:

**Top 5 Most Negative Topics (BERT)**:
1. **Topic 2**: 52.4% negative sentiment (23,526 posts/comments) - **Highest priority**
2. **Topic 6**: 41.1% negative sentiment (16,819 posts/comments)
3. **Topic 7**: 40.8% negative sentiment (33,757 posts/comments) - **Largest volume**
4. **Topic 1**: 40.7% negative sentiment (22,824 posts/comments)
5. **Topic 4**: 34.7% negative sentiment (17,489 posts/comments)

**Top 5 Most Negative Subtopics (BERT)**:
1. **Topic 0 - Subtopic 1**: 56.7% negative (4,155 posts/comments) - **Highest negativity**
2. **Topic 2 - Subtopic 1**: 56.3% negative (3,788 posts/comments)
3. **Topic 2 - Subtopic 2**: 56.0% negative (2,911 posts/comments)
4. **Topic 2 - Subtopic 6**: 54.7% negative (3,441 posts/comments)
5. **Topic 2 - Subtopic 0**: 53.0% negative (3,269 posts/comments)

**Context**: The average negative sentiment across all topics is 34.6%. Topics/subtopics significantly above this average (especially those exceeding 50%) represent high-priority areas requiring urgent intervention.

### Key Findings

1. **Topic-Level Insights**:
   - Topic 2 stands out with over 52% negative sentiment, making it the highest priority for intervention
   - Topic 7, while having 40.8% negative sentiment, has the largest volume (33,757 posts), indicating widespread concern
   - Multiple topics exceed the 40% threshold, suggesting systemic challenges

2. **Subtopic-Level Granularity**:
   - Subtopic analysis reveals specific pain points that may be masked at the topic level
   - Topic 0 - Subtopic 1 shows 56.7% negativity, the highest of all subtopics
   - Topic 2 contains multiple high-negativity subtopics (1, 2, 6, 0), indicating it's a particularly challenging topic area

3. **Model Validation**:
   - BERT and VADER show agreement on the most problematic topics, validating the findings
   - Both models identify Topic 2 as highly negative, though BERT shows higher negative percentages overall

4. **Intervention Priorities**:
   - **Immediate Action**: Topic 2 and Topic 0 - Subtopic 1 require urgent support resources
   - **High Volume Concerns**: Topic 7 needs attention due to its large discussion volume
   - **Targeted Support**: Subtopic-level analysis enables precise intervention strategies

### Supporting Visualizations

1. **`rq3_top_negative_topics_bert.png`** - Horizontal bar chart showing the top 5 topics with highest negative sentiment percentages, with average line for context
2. **`rq3_top_negative_subtopics_bert.png`** - Horizontal bar chart showing the top 8 subtopics with highest negative sentiment, labeled by Topic-Subtopic ID
3. **`rq3_model_comparison_heatmap.png`** - Heatmap comparing BERT vs VADER negative sentiment percentages for top negative topics, validating findings across models
4. **`topic_sentiment_heatmap_bert.png`** & **`subtopic_sentiment_heatmap_bert.png`** (existing) - Comprehensive sentiment distribution matrices showing all topics/subtopics

---

## Summary and Implications

These three research questions provide comprehensive insights into:

1. **Temporal Evolution**: Understanding how discussion topics have shifted over time helps identify emerging concerns, successful interventions, and evolving community needs.

2. **Community Dynamics**: The post-comment sentiment relationship reveals how emotional tone propagates through discussions, informing moderation strategies and community support approaches.

3. **Intervention Priorities**: Identifying high-risk topics and subtopics enables targeted allocation of support resources, clinical attention, and community interventions where they're most needed.

### Recommendations

Based on these findings:

- **For Healthcare Providers**: Focus on Topic 2 and high-negativity subtopics, which show persistent challenges requiring clinical attention.

- **For Community Moderators**: Leverage the post-comment sentiment relationship by encouraging positive framing in posts to foster supportive discussions.

- **For Researchers**: Monitor topic prevalence shifts to identify emerging concerns and evaluate the effectiveness of public health interventions.

- **For Policymakers**: Prioritize resources for topics showing high negative sentiment and large discussion volumes, particularly Topic 7.

---

**All visualizations and supporting data are available in `outputs/eda/top3_rqs/`**

**Generated on**: Analysis completed with data from 2020-2025, covering 167,150+ posts and comments.

