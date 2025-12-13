# Top 3 Research Questions: Comprehensive Analysis

This document provides in-depth analysis of the three most important research questions, each supported by multiple visualizations.

---

**RQ1: Which topics have gained or lost prominence in menopause discussions over time, and how do these shifts relate to changing public health priorities or community needs?**

Based on comprehensive temporal analysis of topic prevalence:

**Top Gaining Topics**: Topic 0 (+10.5%), Topic 6 (+10.4%), Topic 4 (+7.2%)

**Top Losing Topics**: Topic 2 (-4.1%), Topic 1 (-6.1%), Topic 7 (-26.5%)

**Key Findings**:
• The topic prevalence timeline (stacked area chart) reveals significant shifts in discussion focus over time.
• Some topics show steady growth, potentially reflecting increased awareness, treatment access, or emerging concerns.
• Declining topics may indicate resolved issues, shifting priorities, or information saturation.
• These patterns align with evolving public health priorities and community needs.

**Supporting Visualizations**:
1. `rq1_topic_prevalence_timeline.png` - Shows monthly topic distribution over time
2. `rq1_topic_gains_losses.png` - Highlights top gaining and losing topics
3. `topic_prevalence_over_time.png` (existing) - Comprehensive temporal view
4. `topic_sentiment_heatmap_bert.png` & `topic_sentiment_heatmap_vader.png` - Show current sentiment by topic


---

**RQ2: How does the sentiment of a post affect the sentiment of its comments?**

Based on correlation analysis between post sentiment and average comment sentiment:

**BERT Analysis**:
• Correlation coefficient: r = 0.426
• Statistical significance: p = 0.0000
• There is a moderate positive correlation, indicating that posts with more positive sentiment tend to receive comments with more positive sentiment (and vice versa).

**VADER Analysis**:
• Correlation coefficient: r = 0.287
• Statistical significance: p = 0.0001

**Key Findings**:
• The scatter plots show the relationship between post sentiment scores and average comment sentiment scores.
• The sentiment alignment matrix reveals how often comment sentiment matches post sentiment.
• Positive posts tend to attract supportive comments, while negative posts may receive empathetic or problem-solving responses.
• This suggests the Reddit community exhibits emotional contagion or supportive response patterns.

**Supporting Visualizations**:
1. `rq2_post_comment_bert_scatter.png` - Scatter plot with regression line (BERT)
2. `rq2_post_comment_vader_scatter.png` - Scatter plot with regression line (VADER)
3. `rq2_sentiment_alignment_matrix.png` - Shows alignment between post and comment sentiment categories
4. `sentiment_over_time_bert.png` & `sentiment_over_time_vader.png` (existing) - Context for overall sentiment trends


---

**RQ3: Which topics and subtopics show the highest negative sentiment, and what does this reveal about areas requiring urgent intervention or support?**

Based on comprehensive analysis of negative sentiment across topics and subtopics:

**Top Negative Topics (BERT)**:
1. Topic 2: 52.4% negative (23,526 posts/comments)
2. Topic 6: 41.1% negative (16,819 posts/comments)
3. Topic 7: 40.8% negative (33,757 posts/comments)
4. Topic 1: 40.7% negative (22,824 posts/comments)
5. Topic 4: 34.7% negative (17,489 posts/comments)

**Top Negative Subtopics (BERT)**:
1. Topic 0 - Subtopic 1: 56.7% negative (4,155 posts/comments)
2. Topic 2 - Subtopic 1: 56.3% negative (3,788 posts/comments)
3. Topic 2 - Subtopic 2: 56.0% negative (2,911 posts/comments)
4. Topic 2 - Subtopic 6: 54.7% negative (3,441 posts/comments)
5. Topic 2 - Subtopic 0: 53.0% negative (3,269 posts/comments)

**Context**: Average negative sentiment across all topics is 34.6%. Topics/subtopics significantly above this average represent high-priority areas for intervention.

**Key Findings**:
• Certain topics consistently show high negative sentiment, indicating persistent challenges or unmet needs.
• Subtopic analysis reveals granular pain points that may be masked at the topic level.
• BERT and VADER show agreement on the most problematic topics, validating the findings.
• These high-risk areas should be prioritized for support resources, clinical attention, and community interventions.

**Supporting Visualizations**:
1. `rq3_top_negative_topics_bert.png` - Bar chart of top 5 most negative topics
2. `rq3_top_negative_subtopics_bert.png` - Bar chart of top 8 most negative subtopics
3. `rq3_model_comparison_heatmap.png` - BERT vs VADER comparison for validation
4. `topic_sentiment_heatmap_bert.png` & `subtopic_sentiment_heatmap_bert.png` (existing) - Comprehensive sentiment distribution


---

## Summary

These three research questions provide comprehensive insights into:
1. **Temporal Evolution**: How discussion topics have shifted over time
2. **Community Dynamics**: How post sentiment influences comment sentiment
3. **Intervention Priorities**: Which topics/subtopics require urgent support

All visualizations and supporting data are available in `outputs/eda/top3_rqs/`
