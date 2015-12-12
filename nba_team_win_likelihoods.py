from scipy.stats import beta, binom
from scipy import integrate
from matplotlib.ticker import FuncFormatter
import matplotlib
import matplotlib.pyplot as plt
import os

#simple formatter to turn stuff into percentages
def to_percent(y, position):
    return '%.0f%%' % (100*y)
fmt = FuncFormatter((to_percent))

class Team(object):
    def __init__(self, wins, losses, prior_wins = 1, prior_losses = 1, name = None, fc = None, ec = None):
        self.wins = wins
        self.losses = losses
        self.prior_wins = prior_wins
        self.prior_losses = prior_losses
        self.rng = list(range(self.wins, 82 - self.losses + 1))
        self.spread = [self.prob_n_w(N) for N in self.rng]
        if name:
            self.name = name
        if fc:
            self.fc = fc
        if ec:
            self.ec = ec
        
        self.expected_wins = sum([p*w for p,w in zip(self.spread, self.rng)])
    
    def _prob_n_wins(self, true_p, N):
        """ assuming a true win-rate, what is the probability of N wins """
        return binom.pmf(N-self.wins, 82-self.wins-self.losses, true_p)
    
    def _fn(self,r, n_wins):
        """ Function to integrate when calculating probability of N wins.
        Note: posterior distribution of beta distribution is used (also a beta distribution) """
        return beta.pdf(r, self.wins+self.prior_wins, self.losses+self.prior_losses)*self._prob_n_wins(r, n_wins)
    
    def prob_n_w(self, N):
        """ Returns simple probability of N wins """
        return integrate.quad(lambda x: self._fn(x, N), 0, 1)[0]
    
    def win_spread(self):
        """ For each possible outcome (# games won), assign a probability """
        if self.rng:
            return self.rng, self.spread
        else: 
            self.rng = range(self.wins, 82 - self.losses + 1)
            self.spread = [self.prob_n_w(N) for N in self.rng]
            return self.rng, self.spread
    
    def plot_spread(self, ax, xlim = None, ylim = None, **kwargs):
        """ makes a basic plot....more advanced stuff in main method """
        ax.bar(self.rng, self.spread, width = 1, edgecolor = self.ec, facecolor = self.fc, **kwargs)
        return matplotlib.patches.Rectangle((0,0), 2,1,facecolor = self.fc, edgecolor = self.ec, label = self.name) 

if __name__ == "__main__":
    Warriors = Team(24,0,67,15, name = 'Warriors', fc = '#fdb927', ec = '#006bb6')
    Sixers = Team(1,22,18,64, name = '76ers', fc = '#006bb6', ec = '#ed174c')
    teams = Warriors, Sixers
    handles = []
    plt.xkcd()
    fig,ax = plt.subplots()
    for t in teams:
        handles.append(t.plot_spread(ax))
        ax.axvline(x = t.expected_wins, color = t.ec, ls = '--', ymax = max(t.spread)/0.14)
        ax.text(t.expected_wins, max(t.spread) + 0.002, 'Expected: %.1f' % t.expected_wins,
                color = t.ec, va = 'bottom', ha = 'center', fontsize = 'small')
    ax.set_xlim(0,82)
    ax.set_ylim(0,0.14)
    ax.legend(handles = handles, fancybox = True, loc = 'center', fontsize = 'small')
    ax.set_title("Warriors vs Sixers Projected Wins 2015/16")
    for ymaj in ax.yaxis.get_majorticklocs():
        ax.axhline(y = ymaj, ls = '--', color = '#d8d5d5', linewidth = 0.5)
    ax.yaxis.set_major_formatter(fmt)
    ax.tick_params(axis=u'both', which=u'both',length=0)
    ax.set_axis_bgcolor('#d8d5d5')
    ax.set_xlabel('# Wins')
    ax.set_ylabel('Likelihood')
    filename = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'Warriors_vs_Sixers.png')
    #plt.tight_layout()
    plt.savefig(filename, facecolor = '#bec0c2')

