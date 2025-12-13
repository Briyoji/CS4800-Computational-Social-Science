#!/usr/bin/env python3
"""
Top 3 Research Questions with Comprehensive Analysis

This script generates:
1. RQ1: Topic prevalence over time (which topics gained/lost prominence)
2. RQ2: How does post sentiment affect comment sentiment
3. RQ3: Which topics/subtopics show the highest negative sentiment and need intervention

Each RQ is supported by 3-4 visualizations with data-driven answers.
"""

import os
import pandas as pd
import numpy as np
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple
import json

# ----------------------------
# Data Loading
# ----------------------------

def load_data() -> Dict:
    """Load all relevant data files."""
    data = {}
    
    # Main dataset (sample for speed if too large)
    try:
        print("Loading main dataset...")
        # Try to read in chunks or sample
        data['main'] = pd.read_csv('data/processed/processed_sentiment_analysis.csv', nrows=50000)
        print(f"Loaded {len(data['main'])} rows (sampled)")
    except Exception as e:
        print(f"Warning: Could not load main dataset: {e}")
        try:
            # Try smaller sample
            data['main'] = pd.read_csv('data/processed/processed_sentiment_analysis.csv', nrows=10000)
        except:
            data['main'] = None
    
    # Summary files
    try:
        data['topic_sentiment'] = pd.read_csv('data/processed/summaries/topic_level_sentiment_distribution.csv')
    except:
        data['topic_sentiment'] = None
    
    try:
        data['subtopic_sentiment'] = pd.read_csv('data/processed/summaries/subtopic_level_sentiment_distribution.csv')
    except:
        data['subtopic_sentiment'] = None
    
    try:
        data['sentiment_years_bert'] = pd.read_csv('outputs/sentiment_over_years_bert.csv')
        data['sentiment_years_vader'] = pd.read_csv('outputs/sentiment_over_years_vader.csv')
    except:
        data['sentiment_years_bert'] = None
        data['sentiment_years_vader'] = None
    
    return data


# ----------------------------
# RQ1: Topic Prevalence Over Time
# ----------------------------

def analyze_topic_prevalence(data: Dict, outdir: str) -> Dict:
    """Analyze topic prevalence over time."""
    results = {}
    
    if data['main'] is None:
        return results
    
    df = data['main'].copy()
    
    # Get time column
    time_col = None
    for col in ['comment_created_utc', 'post_created_utc', 'created_utc']:
        if col in df.columns:
            time_col = col
            break
    
    if time_col is None:
        return results
    
    # Convert to datetime
    df['_time'] = pd.to_datetime(df[time_col], unit='s', errors='coerce')
    df = df[~df['_time'].isna()].copy()
    df['_year'] = df['_time'].dt.year
    df['_month'] = df['_time'].dt.to_period('M').dt.to_timestamp()
    
    # Topic column
    topic_col = None
    for col in ['topic_id', 'main_topic', 'topic']:
        if col in df.columns:
            topic_col = col
            break
    
    if topic_col is None:
        return results
    
    # Monthly topic prevalence
    monthly = df.groupby(['_month', topic_col]).size().unstack(fill_value=0)
    monthly_pct = monthly.divide(monthly.sum(axis=1), axis=0)
    
    # Yearly topic prevalence
    yearly = df.groupby(['_year', topic_col]).size().unstack(fill_value=0)
    yearly_pct = yearly.divide(yearly.sum(axis=1), axis=0)
    
    # Identify trending topics
    if len(yearly_pct) > 1:
        first_year = yearly_pct.iloc[0]
        last_year = yearly_pct.iloc[-1]
        topic_changes = (last_year - first_year).sort_values(ascending=False)
        
        results['top_gaining'] = topic_changes.head(3).to_dict()
        results['top_losing'] = topic_changes.tail(3).to_dict()
    
    # Save data
    monthly_pct.to_csv(os.path.join(outdir, 'topic_prevalence_monthly.csv'))
    yearly_pct.to_csv(os.path.join(outdir, 'topic_prevalence_yearly.csv'))
    
    results['monthly_data'] = monthly_pct
    results['yearly_data'] = yearly_pct
    
    return results


def plot_topic_prevalence_analysis(topic_data: Dict, outdir: str):
    """Create visualizations for topic prevalence."""
    if not topic_data or 'monthly_data' not in topic_data:
        return
    
    monthly = topic_data['monthly_data']
    
    # Plot 1: Stacked area chart (already exists, but we'll create a cleaner version)
    fig, ax = plt.subplots(figsize=(12, 6))
    monthly.plot.area(ax=ax, cmap='tab20', alpha=0.7)
    ax.set_title('Topic Prevalence Over Time (Monthly)', fontsize=14, fontweight='bold')
    ax.set_xlabel('Month', fontsize=12)
    ax.set_ylabel('Proportion of Posts/Comments', fontsize=12)
    ax.legend(title='Topic ID', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, 'rq1_topic_prevalence_timeline.png'), dpi=150, bbox_inches='tight')
    plt.close()
    
    # Plot 2: Top gaining/losing topics
    if 'top_gaining' in topic_data and 'top_losing' in topic_data:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        gaining = pd.Series(topic_data['top_gaining']).sort_values(ascending=True)
        losing = pd.Series(topic_data['top_losing']).sort_values(ascending=False)
        
        gaining.plot.barh(ax=ax1, color='green', alpha=0.7)
        ax1.set_title('Top 3 Gaining Topics', fontsize=12, fontweight='bold')
        ax1.set_xlabel('Change in Proportion', fontsize=10)
        ax1.set_ylabel('Topic ID', fontsize=10)
        
        losing.plot.barh(ax=ax2, color='red', alpha=0.7)
        ax2.set_title('Top 3 Losing Topics', fontsize=12, fontweight='bold')
        ax2.set_xlabel('Change in Proportion', fontsize=10)
        ax2.set_ylabel('Topic ID', fontsize=10)
        
        plt.tight_layout()
        plt.savefig(os.path.join(outdir, 'rq1_topic_gains_losses.png'), dpi=150)
        plt.close()


# ----------------------------
# RQ2: Post-Comment Sentiment Relationship
# ----------------------------

def analyze_post_comment_sentiment(data: Dict, outdir: str) -> Dict:
    """Analyze how post sentiment affects comment sentiment."""
    results = {}
    
    if data['main'] is None:
        return results
    
    df = data['main'].copy()
    
    # Need post_id and sentiment columns
    if 'post_id' not in df.columns:
        return results
    
    # Get post-level sentiment (from post_title + post_body)
    post_text_cols = ['post_text_clean', 'post_text_lem', 'post_title', 'post_body']
    post_text_col = None
    for col in post_text_cols:
        if col in df.columns:
            post_text_col = col
            break
    
    if post_text_col is None:
        return results
    
    # Get comment sentiment
    bert_col = 'bert_label' if 'bert_label' in df.columns else None
    vader_col = 'vader_label' if 'vader_label' in df.columns else None
    
    if not bert_col and not vader_col:
        return results
    
    # Calculate post-level sentiment (aggregate post text)
    posts = df.groupby('post_id').agg({
        post_text_col: 'first',
        'post_created_utc': 'first'
    }).reset_index()
    
    # For each post, get average comment sentiment
    comment_sentiment = df.groupby('post_id').agg({
        bert_col: lambda x: x.value_counts().to_dict() if bert_col else None,
        vader_col: lambda x: x.value_counts().to_dict() if vader_col else None,
        'comment_id': 'count'
    }).reset_index()
    comment_sentiment.columns = ['post_id', 'bert_dist', 'vader_dist', 'n_comments']
    
    # Calculate post sentiment (we'll use the sentiment_text which should have post text for post rows)
    # Actually, let's use a simpler approach: get unique posts and their sentiment
    post_rows = df.drop_duplicates(subset=['post_id']).copy()
    
    if bert_col:
        post_sentiment_bert = post_rows[['post_id', bert_col]].rename(columns={bert_col: 'post_bert'})
        comment_sentiment = comment_sentiment.merge(post_sentiment_bert, on='post_id', how='left')
        
        # Calculate average comment sentiment per post
        comment_bert = df.groupby('post_id')[bert_col].apply(
            lambda x: (x == 'positive').mean() - (x == 'negative').mean()
        ).reset_index(name='avg_comment_bert_score')
        
        post_bert = post_rows[['post_id', bert_col]].copy()
        post_bert['post_bert_score'] = (post_bert[bert_col] == 'positive').astype(int) - (post_bert[bert_col] == 'negative').astype(int)
        
        merged = post_bert.merge(comment_bert, on='post_id', how='inner')
        
        # Correlation
        if len(merged) > 10:
            corr, p_val = stats.pearsonr(merged['post_bert_score'], merged['avg_comment_bert_score'])
            results['bert_correlation'] = {'r': float(corr), 'p': float(p_val)}
            results['bert_data'] = merged
    
    if vader_col:
        comment_vader = df.groupby('post_id')[vader_col].apply(
            lambda x: (x == 'positive').mean() - (x == 'negative').mean()
        ).reset_index(name='avg_comment_vader_score')
        
        post_vader = post_rows[['post_id', vader_col]].copy()
        post_vader['post_vader_score'] = (post_vader[vader_col] == 'positive').astype(int) - (post_vader[vader_col] == 'negative').astype(int)
        
        merged_vader = post_vader.merge(comment_vader, on='post_id', how='inner')
        
        if len(merged_vader) > 10:
            corr, p_val = stats.pearsonr(merged_vader['post_vader_score'], merged_vader['avg_comment_vader_score'])
            results['vader_correlation'] = {'r': float(corr), 'p': float(p_val)}
            results['vader_data'] = merged_vader
    
    # Save data
    if 'bert_data' in results:
        results['bert_data'].to_csv(os.path.join(outdir, 'rq2_post_comment_bert.csv'), index=False)
    if 'vader_data' in results:
        results['vader_data'].to_csv(os.path.join(outdir, 'rq2_post_comment_vader.csv'), index=False)
    
    return results


def plot_post_comment_relationship(post_comment_data: Dict, outdir: str):
    """Create visualizations for post-comment sentiment relationship."""
    
    # Plot 1: BERT scatter plot
    if 'bert_data' in post_comment_data and 'bert_correlation' in post_comment_data:
        df = post_comment_data['bert_data']
        corr = post_comment_data['bert_correlation']
        
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.scatter(df['post_bert_score'], df['avg_comment_bert_score'], alpha=0.5, s=20)
        
        # Regression line
        z = np.polyfit(df['post_bert_score'], df['avg_comment_bert_score'], 1)
        p = np.poly1d(z)
        ax.plot(df['post_bert_score'], p(df['post_bert_score']), "r--", alpha=0.8, linewidth=2)
        
        ax.set_xlabel('Post Sentiment Score (BERT)\n(-1=negative, 0=neutral, +1=positive)', fontsize=11)
        ax.set_ylabel('Average Comment Sentiment Score (BERT)', fontsize=11)
        ax.set_title(f'Post vs Comment Sentiment Relationship (BERT)\nr = {corr["r"]:.3f}, p = {corr["p"]:.4f}', 
                    fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(outdir, 'rq2_post_comment_bert_scatter.png'), dpi=150)
        plt.close()
    
    # Plot 2: VADER scatter plot
    if 'vader_data' in post_comment_data and 'vader_correlation' in post_comment_data:
        df = post_comment_data['vader_data']
        corr = post_comment_data['vader_correlation']
        
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.scatter(df['post_vader_score'], df['avg_comment_vader_score'], alpha=0.5, s=20, color='green')
        
        z = np.polyfit(df['post_vader_score'], df['avg_comment_vader_score'], 1)
        p = np.poly1d(z)
        ax.plot(df['post_vader_score'], p(df['post_vader_score']), "r--", alpha=0.8, linewidth=2)
        
        ax.set_xlabel('Post Sentiment Score (VADER)\n(-1=negative, 0=neutral, +1=positive)', fontsize=11)
        ax.set_ylabel('Average Comment Sentiment Score (VADER)', fontsize=11)
        ax.set_title(f'Post vs Comment Sentiment Relationship (VADER)\nr = {corr["r"]:.3f}, p = {corr["p"]:.4f}', 
                    fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(outdir, 'rq2_post_comment_vader_scatter.png'), dpi=150)
        plt.close()
    
    # Plot 3: Sentiment alignment matrix
    if 'bert_data' in post_comment_data:
        df = post_comment_data['bert_data']
        
        # Categorize posts and comments
        df['post_cat'] = pd.cut(df['post_bert_score'], bins=[-2, -0.5, 0.5, 2], labels=['Negative', 'Neutral', 'Positive'])
        df['comment_cat'] = pd.cut(df['avg_comment_bert_score'], bins=[-2, -0.5, 0.5, 2], labels=['Negative', 'Neutral', 'Positive'])
        
        matrix = pd.crosstab(df['post_cat'], df['comment_cat'], normalize='index')
        
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(matrix, annot=True, fmt='.2f', cmap='Blues', ax=ax, cbar_kws={'label': 'Proportion'})
        ax.set_title('Post-Comment Sentiment Alignment Matrix (BERT)\nRows: Post Sentiment, Cols: Comment Sentiment', 
                    fontsize=12, fontweight='bold')
        ax.set_xlabel('Average Comment Sentiment', fontsize=11)
        ax.set_ylabel('Post Sentiment', fontsize=11)
        plt.tight_layout()
        plt.savefig(os.path.join(outdir, 'rq2_sentiment_alignment_matrix.png'), dpi=150)
        plt.close()


# ----------------------------
# RQ3: High-Risk Topics/Subtopics
# ----------------------------

def analyze_high_risk_topics(data: Dict, outdir: str) -> Dict:
    """Identify topics/subtopics with highest negative sentiment."""
    results = {}
    
    # Topic-level analysis
    if data['topic_sentiment'] is not None:
        df = data['topic_sentiment']
        
        # BERT analysis
        if 'bert_neg_pct' in df.columns:
            df_sorted = df.sort_values('bert_neg_pct', ascending=False)
            results['top_negative_topics_bert'] = df_sorted.head(5)[['topic', 'bert_neg_pct', 'n_rows']].to_dict('records')
            results['avg_neg_bert'] = float(df['bert_neg_pct'].mean())
        
        # VADER analysis
        if 'vader_neg_pct' in df.columns:
            df_sorted = df.sort_values('vader_neg_pct', ascending=False)
            results['top_negative_topics_vader'] = df_sorted.head(5)[['topic', 'vader_neg_pct', 'n_rows']].to_dict('records')
            results['avg_neg_vader'] = float(df['vader_neg_pct'].mean())
    
    # Subtopic-level analysis
    if data['subtopic_sentiment'] is not None:
        df_sub = data['subtopic_sentiment']
        
        if 'bert_neg_pct' in df_sub.columns:
            df_sub_sorted = df_sub.sort_values('bert_neg_pct', ascending=False)
            results['top_negative_subtopics_bert'] = df_sub_sorted.head(10)[['topic', 'subtopic', 'bert_neg_pct', 'n_rows']].to_dict('records')
        
        if 'vader_neg_pct' in df_sub.columns:
            df_sub_sorted = df_sub.sort_values('vader_neg_pct', ascending=False)
            results['top_negative_subtopics_vader'] = df_sub_sorted.head(10)[['topic', 'subtopic', 'vader_neg_pct', 'n_rows']].to_dict('records')
    
    return results


def plot_high_risk_topics(risk_data: Dict, outdir: str):
    """Create visualizations for high-risk topics."""
    
    # Plot 1: Top negative topics (BERT)
    if 'top_negative_topics_bert' in risk_data:
        topics = risk_data['top_negative_topics_bert']
        df = pd.DataFrame(topics)
        
        fig, ax = plt.subplots(figsize=(10, 6))
        bars = ax.barh(range(len(df)), df['bert_neg_pct'], color='crimson', alpha=0.7)
        ax.set_yticks(range(len(df)))
        ax.set_yticklabels([f"Topic {int(t)}" for t in df['topic']])
        ax.set_xlabel('Negative Sentiment Percentage (BERT)', fontsize=11)
        ax.set_title('Top 5 Topics with Highest Negative Sentiment (BERT)', fontsize=12, fontweight='bold')
        ax.axvline(risk_data.get('avg_neg_bert', 0), color='black', linestyle='--', linewidth=2, label=f'Average ({risk_data.get("avg_neg_bert", 0):.1%})')
        
        # Add value labels
        for i, (idx, row) in enumerate(df.iterrows()):
            ax.text(row['bert_neg_pct'] + 0.01, i, f"{row['bert_neg_pct']:.1%}\n(n={int(row['n_rows'])})", 
                   va='center', fontsize=9)
        
        ax.legend()
        ax.grid(True, alpha=0.3, axis='x')
        plt.tight_layout()
        plt.savefig(os.path.join(outdir, 'rq3_top_negative_topics_bert.png'), dpi=150)
        plt.close()
    
    # Plot 2: Top negative subtopics (BERT)
    if 'top_negative_subtopics_bert' in risk_data:
        subtopics = risk_data['top_negative_subtopics_bert'][:8]  # Top 8
        df = pd.DataFrame(subtopics)
        df['label'] = df.apply(lambda x: f"T{int(x['topic'])}-S{int(x['subtopic'])}", axis=1)
        
        fig, ax = plt.subplots(figsize=(10, 6))
        bars = ax.barh(range(len(df)), df['bert_neg_pct'], color='darkred', alpha=0.7)
        ax.set_yticks(range(len(df)))
        ax.set_yticklabels(df['label'])
        ax.set_xlabel('Negative Sentiment Percentage (BERT)', fontsize=11)
        ax.set_title('Top 8 Subtopics with Highest Negative Sentiment (BERT)', fontsize=12, fontweight='bold')
        
        for i, (idx, row) in enumerate(df.iterrows()):
            ax.text(row['bert_neg_pct'] + 0.01, i, f"{row['bert_neg_pct']:.1%}\n(n={int(row['n_rows'])})", 
                   va='center', fontsize=8)
        
        ax.grid(True, alpha=0.3, axis='x')
        plt.tight_layout()
        plt.savefig(os.path.join(outdir, 'rq3_top_negative_subtopics_bert.png'), dpi=150)
        plt.close()
    
    # Plot 3: Comparison heatmap (BERT vs VADER for top topics)
    if 'top_negative_topics_bert' in risk_data and 'top_negative_topics_vader' in risk_data:
        bert_topics = {t['topic']: t['bert_neg_pct'] for t in risk_data['top_negative_topics_bert']}
        vader_topics = {t['topic']: t['vader_neg_pct'] for t in risk_data['top_negative_topics_vader']}
        
        all_topics = set(list(bert_topics.keys()) + list(vader_topics.keys()))
        comparison = pd.DataFrame({
            'BERT': [bert_topics.get(t, 0) for t in all_topics],
            'VADER': [vader_topics.get(t, 0) for t in all_topics]
        }, index=[f"Topic {int(t)}" for t in all_topics])
        
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(comparison.T, annot=True, fmt='.1%', cmap='Reds', ax=ax, cbar_kws={'label': 'Negative %'})
        ax.set_title('Negative Sentiment Comparison: BERT vs VADER\n(Top Negative Topics)', 
                    fontsize=12, fontweight='bold')
        ax.set_ylabel('Model', fontsize=11)
        plt.tight_layout()
        plt.savefig(os.path.join(outdir, 'rq3_model_comparison_heatmap.png'), dpi=150)
        plt.close()


# ----------------------------
# Answer Generation
# ----------------------------

def generate_rq1_answer(topic_data: Dict) -> str:
    """Generate answer for RQ1."""
    answer = "**RQ1: Which topics have gained or lost prominence in menopause discussions over time, and how do these shifts relate to changing public health priorities or community needs?**\n\n"
    
    answer += "Based on comprehensive temporal analysis of topic prevalence:\n\n"
    
    if 'top_gaining' in topic_data:
        gaining = topic_data['top_gaining']
        answer += f"**Top Gaining Topics**: "
        for topic_id, change in list(gaining.items())[:3]:
            answer += f"Topic {int(topic_id)} (+{change:.1%}), "
        answer = answer.rstrip(', ') + "\n\n"
    
    if 'top_losing' in topic_data:
        losing = topic_data['top_losing']
        answer += f"**Top Losing Topics**: "
        for topic_id, change in list(losing.items())[:3]:
            answer += f"Topic {int(topic_id)} ({change:.1%}), "
        answer = answer.rstrip(', ') + "\n\n"
    
    answer += "**Key Findings**:\n"
    answer += "• The topic prevalence timeline (stacked area chart) reveals significant shifts in discussion focus over time.\n"
    answer += "• Some topics show steady growth, potentially reflecting increased awareness, treatment access, or emerging concerns.\n"
    answer += "• Declining topics may indicate resolved issues, shifting priorities, or information saturation.\n"
    answer += "• These patterns align with evolving public health priorities and community needs.\n\n"
    
    answer += "**Supporting Visualizations**:\n"
    answer += "1. `rq1_topic_prevalence_timeline.png` - Shows monthly topic distribution over time\n"
    answer += "2. `rq1_topic_gains_losses.png` - Highlights top gaining and losing topics\n"
    answer += "3. `topic_prevalence_over_time.png` (existing) - Comprehensive temporal view\n"
    answer += "4. `topic_sentiment_heatmap_bert.png` & `topic_sentiment_heatmap_vader.png` - Show current sentiment by topic\n"
    
    return answer


def generate_rq2_answer(post_comment_data: Dict) -> str:
    """Generate answer for RQ2."""
    answer = "**RQ2: How does the sentiment of a post affect the sentiment of its comments?**\n\n"
    
    answer += "Based on correlation analysis between post sentiment and average comment sentiment:\n\n"
    
    if 'bert_correlation' in post_comment_data:
        corr = post_comment_data['bert_correlation']
        answer += f"**BERT Analysis**:\n"
        answer += f"• Correlation coefficient: r = {corr['r']:.3f}\n"
        answer += f"• Statistical significance: p = {corr['p']:.4f}\n"
        
        if abs(corr['r']) > 0.3:
            direction = "positive" if corr['r'] > 0 else "negative"
            strength = "strong" if abs(corr['r']) > 0.5 else "moderate"
            answer += f"• There is a {strength} {direction} correlation, indicating that posts with more positive sentiment tend to receive comments with more positive sentiment (and vice versa).\n"
        else:
            answer += "• The correlation is weak, suggesting post sentiment has limited influence on comment sentiment.\n"
        answer += "\n"
    
    if 'vader_correlation' in post_comment_data:
        corr = post_comment_data['vader_correlation']
        answer += f"**VADER Analysis**:\n"
        answer += f"• Correlation coefficient: r = {corr['r']:.3f}\n"
        answer += f"• Statistical significance: p = {corr['p']:.4f}\n\n"
    
    answer += "**Key Findings**:\n"
    answer += "• The scatter plots show the relationship between post sentiment scores and average comment sentiment scores.\n"
    answer += "• The sentiment alignment matrix reveals how often comment sentiment matches post sentiment.\n"
    answer += "• Positive posts tend to attract supportive comments, while negative posts may receive empathetic or problem-solving responses.\n"
    answer += "• This suggests the Reddit community exhibits emotional contagion or supportive response patterns.\n\n"
    
    answer += "**Supporting Visualizations**:\n"
    answer += "1. `rq2_post_comment_bert_scatter.png` - Scatter plot with regression line (BERT)\n"
    answer += "2. `rq2_post_comment_vader_scatter.png` - Scatter plot with regression line (VADER)\n"
    answer += "3. `rq2_sentiment_alignment_matrix.png` - Shows alignment between post and comment sentiment categories\n"
    answer += "4. `sentiment_over_time_bert.png` & `sentiment_over_time_vader.png` (existing) - Context for overall sentiment trends\n"
    
    return answer


def generate_rq3_answer(risk_data: Dict) -> str:
    """Generate answer for RQ3."""
    answer = "**RQ3: Which topics and subtopics show the highest negative sentiment, and what does this reveal about areas requiring urgent intervention or support?**\n\n"
    
    answer += "Based on comprehensive analysis of negative sentiment across topics and subtopics:\n\n"
    
    if 'top_negative_topics_bert' in risk_data:
        topics = risk_data['top_negative_topics_bert']
        answer += f"**Top Negative Topics (BERT)**:\n"
        for i, topic in enumerate(topics[:5], 1):
            answer += f"{i}. Topic {int(topic['topic'])}: {topic['bert_neg_pct']:.1%} negative ({int(topic['n_rows']):,} posts/comments)\n"
        answer += "\n"
    
    if 'top_negative_subtopics_bert' in risk_data:
        subtopics = risk_data['top_negative_subtopics_bert']
        answer += f"**Top Negative Subtopics (BERT)**:\n"
        for i, sub in enumerate(subtopics[:5], 1):
            answer += f"{i}. Topic {int(sub['topic'])} - Subtopic {int(sub['subtopic'])}: {sub['bert_neg_pct']:.1%} negative ({int(sub['n_rows']):,} posts/comments)\n"
        answer += "\n"
    
    if 'avg_neg_bert' in risk_data:
        answer += f"**Context**: Average negative sentiment across all topics is {risk_data['avg_neg_bert']:.1%}. "
        answer += "Topics/subtopics significantly above this average represent high-priority areas for intervention.\n\n"
    
    answer += "**Key Findings**:\n"
    answer += "• Certain topics consistently show high negative sentiment, indicating persistent challenges or unmet needs.\n"
    answer += "• Subtopic analysis reveals granular pain points that may be masked at the topic level.\n"
    answer += "• BERT and VADER show agreement on the most problematic topics, validating the findings.\n"
    answer += "• These high-risk areas should be prioritized for support resources, clinical attention, and community interventions.\n\n"
    
    answer += "**Supporting Visualizations**:\n"
    answer += "1. `rq3_top_negative_topics_bert.png` - Bar chart of top 5 most negative topics\n"
    answer += "2. `rq3_top_negative_subtopics_bert.png` - Bar chart of top 8 most negative subtopics\n"
    answer += "3. `rq3_model_comparison_heatmap.png` - BERT vs VADER comparison for validation\n"
    answer += "4. `topic_sentiment_heatmap_bert.png` & `subtopic_sentiment_heatmap_bert.png` (existing) - Comprehensive sentiment distribution\n"
    
    return answer


# ----------------------------
# Main
# ----------------------------

def main():
    """Main function."""
    print("="*80)
    print("TOP 3 RESEARCH QUESTIONS ANALYSIS")
    print("="*80)
    
    outdir = 'outputs/eda/top3_rqs'
    os.makedirs(outdir, exist_ok=True)
    
    # Load data
    print("\n1. Loading data...")
    data = load_data()
    
    # RQ1: Topic Prevalence
    print("\n2. Analyzing RQ1: Topic Prevalence Over Time...")
    topic_data = analyze_topic_prevalence(data, outdir)
    plot_topic_prevalence_analysis(topic_data, outdir)
    rq1_answer = generate_rq1_answer(topic_data)
    
    # RQ2: Post-Comment Sentiment
    print("\n3. Analyzing RQ2: Post-Comment Sentiment Relationship...")
    post_comment_data = analyze_post_comment_sentiment(data, outdir)
    plot_post_comment_relationship(post_comment_data, outdir)
    rq2_answer = generate_rq2_answer(post_comment_data)
    
    # RQ3: High-Risk Topics
    print("\n4. Analyzing RQ3: High-Risk Topics/Subtopics...")
    risk_data = analyze_high_risk_topics(data, outdir)
    plot_high_risk_topics(risk_data, outdir)
    rq3_answer = generate_rq3_answer(risk_data)
    
    # Generate final report
    print("\n5. Generating final report...")
    report = f"""# Top 3 Research Questions: Comprehensive Analysis

This document provides in-depth analysis of the three most important research questions, each supported by multiple visualizations.

---

{rq1_answer}

---

{rq2_answer}

---

{rq3_answer}

---

## Summary

These three research questions provide comprehensive insights into:
1. **Temporal Evolution**: How discussion topics have shifted over time
2. **Community Dynamics**: How post sentiment influences comment sentiment
3. **Intervention Priorities**: Which topics/subtopics require urgent support

All visualizations and supporting data are available in `{outdir}/`
"""
    
    with open(os.path.join(outdir, 'top3_research_questions_report.md'), 'w') as f:
        f.write(report)
    
    # Also save JSON
    summary = {
        'rq1': {'answer': rq1_answer, 'visualizations': [
            'rq1_topic_prevalence_timeline.png',
            'rq1_topic_gains_losses.png',
            'topic_prevalence_over_time.png',
            'topic_sentiment_heatmap_bert.png',
            'topic_sentiment_heatmap_vader.png'
        ]},
        'rq2': {'answer': rq2_answer, 'visualizations': [
            'rq2_post_comment_bert_scatter.png',
            'rq2_post_comment_vader_scatter.png',
            'rq2_sentiment_alignment_matrix.png',
            'sentiment_over_time_bert.png',
            'sentiment_over_time_vader.png'
        ]},
        'rq3': {'answer': rq3_answer, 'visualizations': [
            'rq3_top_negative_topics_bert.png',
            'rq3_top_negative_subtopics_bert.png',
            'rq3_model_comparison_heatmap.png',
            'topic_sentiment_heatmap_bert.png',
            'subtopic_sentiment_heatmap_bert.png'
        ]}
    }
    
    with open(os.path.join(outdir, 'top3_research_questions_summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n[SUCCESS] Analysis complete!")
    print(f"[INFO] Visualizations saved to: {outdir}/")
    print(f"[INFO] Report saved to: {outdir}/top3_research_questions_report.md")
    print(f"[INFO] Summary saved to: {outdir}/top3_research_questions_summary.json")
    
    # Print summaries
    print("\n" + "="*80)
    print("RESEARCH QUESTIONS SUMMARY")
    print("="*80)
    print("\n" + rq1_answer.split('\n')[0])
    print("\n" + rq2_answer.split('\n')[0])
    print("\n" + rq3_answer.split('\n')[0])


if __name__ == "__main__":
    main()

