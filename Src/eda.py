import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

class EDA:
    """
    Class for performing Exploratory Data Analysis on the NBA dataset.
    """

    def __init__(self, data):
        """
        Initialize NBAExploratoryDataAnalysis object with NBA dataset.

        Parameters:
        - data (DataFrame): NBA dataset
        """
        self.data = data

    def calculate_elo_rating_changes(self):
        """
        Calculate the difference between pre and post ELO ratings for both teams.
        """
        self.data['elo1_diff'] = self.data['elo1_post'] - self.data['elo1_pre']
        self.data['elo2_diff'] = self.data['elo2_post'] - self.data['elo2_pre']

    def plot_elo_rating_changes_distribution(self):
        """
        Plot the distribution of ELO rating changes (pre vs post) for both teams.
        """
        plt.figure(figsize=(12, 6))
        sns.histplot(self.data[['elo1_diff', 'elo2_diff']], kde=True, element='step', bins=30)
        plt.title('Distribution of ELO Rating Changes (Pre vs Post)')
        plt.xlabel('ELO Rating Change')
        plt.ylabel('Frequency')
        plt.legend(['Team 1', 'Team 2'])
        plt.show()

    def plot_correlation_heatmap(self):
        """
        Plot the correlation heatmap between CARMELO and RAPTOR Ratings.
        """
        correlation_matrix = self.data[['carm-elo1_pre', 'carm-elo2_pre', 'raptor1_pre', 'raptor2_pre']].corr()

        plt.figure(figsize=(10, 8))
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0)
        plt.title('Correlation between CARMELO and RAPTOR Ratings')
        plt.show()


    def plot_elo_probabilities_vs_outcomes(self):
        """
        Plot ELO probabilities against actual outcomes.

        
        Add a column indicating the actual outcomes of the games.

        """
        self.data['outcome'] = np.where(self.data['score1'] > self.data['score2'], 1, 0)
        
        plt.figure(figsize=(12, 6))
        sns.histplot(self.data[self.data['outcome'] == 1]['elo_prob1'], bins=30, kde=True, color='blue', alpha=0.6, label='Winning Team 1')
        sns.histplot(self.data[self.data['outcome'] == 0]['elo_prob1'], bins=30, kde=True, color='red', alpha=0.6, label='Winning Team 2')
        plt.title('ELO Probabilities vs Actual Outcomes')
        plt.xlabel('ELO Probability for Team 1')
        plt.ylabel('Frequency')
        plt.legend()
        plt.show()

    def plot_distribution_of_scores(self):
        """
        Plot the distribution of game scores for both teams using kernel density estimation (KDE) plots.
        """
        plt.figure(figsize=(12, 6))
        sns.kdeplot(self.data['score1'], color='blue', label='Team 1', fill=True)
        sns.kdeplot(self.data['score2'], color='red', label='Team 2', fill=True)
        plt.title('Distribution of Game Scores')
        plt.xlabel('Score')
        plt.ylabel('Density')
        plt.legend()
        plt.show()



    def plot_neutral_vs_non_neutral_games(self):
        """
        Plot comparison of ELO ratings between neutral and non-neutral games.
        """
        plt.figure(figsize=(12, 6))
        sns.boxplot(x='neutral', y='elo1_pre', data=self.data)
        plt.title('ELO Ratings Comparison (Neutral vs Non-neutral Games)')
        plt.xlabel('Neutral Game')
        plt.ylabel('ELO Rating for Team 1')
        plt.show()

    def plot_importance_across_seasons(self):
        """
        Plot the importance of games across seasons.
        """
        plt.figure(figsize=(12, 6))
        sns.boxplot(x='season', y='importance', data=self.data)
        plt.title('Importance of Games across Seasons')
        plt.xlabel('Season')
        plt.ylabel('Importance')
        plt.xticks(rotation=45)
        plt.show()

    def plot_total_rating_vs_quality(self):
        """
        Plot the relationship between total rating and game quality.
        """
        plt.figure(figsize=(12, 6))
        sns.scatterplot(x='total_rating', y='quality', data=self.data)
        plt.title('Total Rating vs Game Quality')
        plt.xlabel('Total Rating')
        plt.ylabel('Quality')
        plt.show()

    def plot_correlation_of_probabilities_with_outcome(self):
        """
        Plot correlation of different probabilities with actual outcomes.
        """
        prob_columns = ['elo_prob1', 'carm-elo_prob1', 'raptor_prob1']
        corr_with_outcome = self.data[prob_columns + ['outcome']].corr()['outcome'][:-1]

        plt.figure(figsize=(12, 6))
        corr_with_outcome.plot(kind='bar')
        plt.title('Correlation of Different Probabilities with Actual Outcomes')
        plt.xlabel('Probability Metric')
        plt.ylabel('Correlation with Actual Outcome')
        plt.show()

    def plot_pre_and_post_ratings_during_playoffs_vs_regular_season(self):
        """
        Plot comparison of pre and post ratings during playoffs vs regular season.
        """
        plt.figure(figsize=(12, 6))
        sns.boxplot(x='playoff', y='elo1_diff', data=self.data)
        plt.title('ELO Rating Changes during Playoff vs Regular Season')
        plt.xlabel('Playoff Game')
        plt.ylabel('ELO Rating Change for Team 1')
        plt.show()

    def Correlation_Between_Pregame_Elo_Ratings_of_Both_Teams (self):

        plt.figure(figsize=(15, 3))
        sns.jointplot(data=self.data, x='elo1_pre', y='elo2_pre', hue='score1', height=8)
        plt.suptitle('Correlation Between Pre-game Elo Ratings of Both Teams', y=1.02)
        plt.xlabel('Team 1 Elo Rating')
        plt.ylabel('Team 2 Elo Rating')
        plt.legend(title='Team 1 Win', loc='upper right')
        plt.show()  
    def  Distribution_of_Game_Quality_Ratings (self):

        plt.figure(figsize=(15, 4))
        sns.histplot(data=self.data, x='quality', bins=30, kde=True)
        plt.title('Distribution of Game Quality Ratings')
        plt.xlabel('Game Quality')
        plt.ylabel('Frequency')
        plt.show()   

    def Ratings_vs_outcome_of_two_teams (self):

        plt.figure(figsize=(12, 6))
        # Plot for Team 1
        plt.subplot(1, 2, 1)
        sns.scatterplot(data=self.data, x='raptor1_pre', y='score1', hue='score1', palette='coolwarm')
        plt.title('Team 1: RAPTOR Rating vs. Game Outcome')
        plt.xlabel('Team 1 RAPTOR Rating')
        plt.ylabel('Game Outcome')
        plt.legend(title='Game Outcome', loc='upper right')

        # Plot for Team 2
        plt.subplot(1, 2, 2)
        sns.scatterplot(data=self.data, x='raptor2_pre', y='score2', hue='score2', palette='plasma')
        plt.title('Team 2: RAPTOR Rating vs. Game Outcome')
        plt.xlabel('Team 2 RAPTOR Rating')
        plt.ylabel('Game Outcome')
        plt.legend(title='Game Outcome', loc='upper right')

        plt.tight_layout()
        plt.show()





