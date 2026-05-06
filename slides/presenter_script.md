# Presenter Script: Friedman TreeBoost

Target length: about 12-15 minutes for 20 slides.

## Slide 1 — Title

Today I am presenting Jerome Friedman's 2001 paper, *Greedy Function Approximation: A Gradient Boosting Machine*.

The main idea is that boosting can be understood as numerical optimization, but instead of optimizing a finite vector of parameters, we optimize a whole prediction function. I also implemented the regression versions from scratch: least-squares TreeBoost, LAD TreeBoost, and Huber, or M, TreeBoost.

The goal of the talk is to explain the statistical method, show how the algorithm works, and then connect it to the implementation and experiments.

Transition: I will start with the roadmap so the argument is easy to follow.

## Slide 2 — Roadmap

The talk has seven parts.

First, I will explain the function-estimation problem and where the paper sits historically. Then I will introduce the key insight: boosting as gradient descent in function space.

After that, I will go through the generic gradient boosting algorithm and its three regression special cases: least squares, LAD, and Huber. The most important technical detail is the terminal-region update, so I will spend a slide on that.

Finally, I will summarize the from-scratch implementation, show the experiments, and close with what worked, what did not, and why the paper matters.

Transition: The starting point is the statistical objective.

## Slide 3 — The Function-Estimation Problem

The paper frames supervised learning as function estimation.

We observe training data \((\mathbf{x}_i, y_i)\), and we want a function \(F(\mathbf{x})\) that minimizes expected loss. The formula on the slide writes this as an optimization over functions, not just over parameters.

Different choices of loss give different learning problems. Squared error gives the classical regression objective. Absolute error gives a more robust regression objective. Logistic loss gives a classification objective.

The important shift in Friedman's paper is that he does not begin by choosing a fixed parametric family for \(F\). Instead, he builds \(F\) stage by stage, using simple weak learners.

Transition: That idea connected several older methods into one framework.

## Slide 4 — Where This Paper Sits

Before this paper, boosting was already known through AdaBoost, and there were related methods like LogitBoost, MARS, neural networks, support vector machines, and other basis-function methods.

But these methods looked more separate than they do now. AdaBoost, for example, was often described through reweighting examples, not as a general optimization procedure.

Friedman's contribution was to unify boosting as gradient descent in function space. Once you see boosting this way, you can plug in different losses and get different algorithms from the same template.

This is also why the paper is historically important. Modern systems like XGBoost, LightGBM, and CatBoost all extend this basic paradigm.

Transition: The mechanism that makes this possible is the stagewise additive model.

## Slide 5 — Stagewise Additive Expansions

Here \(F\) is written as a sum of weak learners.

Each weak learner \(h(\mathbf{x}; \mathbf{a}_m)\) is a simple model, and in TreeBoost it is a small regression tree. The coefficient \(\beta_m\) controls how much that weak learner contributes.

The hard version would be to jointly optimize all weak learners and all coefficients at once. That is usually not tractable.

The greedy stagewise alternative is to freeze everything already chosen, then choose only the next weak learner and its coefficient. This is equation 9 in the paper.

So the model becomes a sequence of small decisions rather than one huge optimization problem.

Transition: The key question is how to choose the next weak learner efficiently.

## Slide 6 — The Key Insight

This is the core idea of the paper.

At the training points, think of the values \(F(\mathbf{x}_i)\) as the current coordinates of the model. If we could freely change each coordinate, the steepest descent direction would be the negative gradient of the loss with respect to \(F(\mathbf{x}_i)\).

Those negative gradients are the pseudo-responses \(\tilde{y}_i\).

But the gradient is only defined at the training points. To generalize it to new inputs, Friedman fits a weak learner to those pseudo-responses. So each new tree is a smooth, structured approximation to the steepest descent direction.

That is why boosting becomes gradient descent in function space.

Transition: The next slide turns that idea into the generic algorithm.

## Slide 7 — Algorithm 1: Generic Gradient Boost

The algorithm starts with a constant prediction \(F_0\), chosen to minimize the empirical loss.

Then for each boosting iteration, line 3 computes the pseudo-responses: the negative gradient of the loss evaluated at the current model.

Line 4 fits a weak learner to those pseudo-responses by least squares. This is important: even if the original loss is not squared error, the tree-fitting step is still a least-squares fit to the current gradient targets.

Line 5 performs a one-dimensional line search in the original loss. Then line 6 updates the function by adding the new weak learner.

The key point is that line 3 and line 5 depend on the chosen loss, while the overall machinery stays the same.

Transition: Now we can specialize this generic algorithm to particular regression losses.

## Slide 8 — Algorithm 2: LS Boost

For squared-error loss, the negative gradient is just the residual \(y_i - F_{m-1}(\mathbf{x}_i)\).

So LS Boost reduces to a familiar recipe: repeatedly fit a regression tree to the current residuals and add it to the model.

The initial constant is the sample mean, and the leaf update is also the mean residual in each leaf.

This is the simplest case, but it is also a useful sanity check. If the general framework does not reduce to residual fitting for squared error, then something is wrong.

Transition: The next loss changes the behavior substantially.

## Slide 9 — Algorithm 3: LAD TreeBoost

For absolute-error loss, the pseudo-response is the sign of the residual, not the residual itself.

This means a residual of 100 and a residual of 1 have the same gradient direction. That is what makes LAD robust to outliers.

But the important part is the leaf update. The tree is fit to signs of residuals, but the final value assigned to each terminal region is the median residual inside that region.

So the algorithm uses least squares to find a useful partition, then uses the original LAD loss to determine the actual leaf values.

Transition: Huber loss sits between LS and LAD.

## Slide 10 — Algorithm 4: M TreeBoost, or Huber TreeBoost

Huber loss is quadratic for small residuals and linear for large residuals.

So for inliers, it behaves like squared error and uses residual magnitudes. For outliers, it clips the gradient and prevents extreme residuals from dominating the model.

The threshold \(\delta_m\) is adaptive: it is set from a quantile of the absolute residuals at each iteration.

The leaf update is more complicated than in LS or LAD. Friedman derives a one-step Newton-like adjustment around the residual median. In the implementation, this is the most technical loss-specific formula.

Transition: This motivates a closer look at the terminal-region update.

## Slide 11 — The Terminal-Region Update

When the weak learner is a tree, each iteration has two stages.

First, build a tree by least squares on the pseudo-responses. This chooses the terminal regions \(R_{jm}\).

Second, for each region, re-solve a tiny one-dimensional problem using the original loss. That gives the region-specific update \(\gamma_{jm}\).

This is the practical heart of TreeBoost. The tree gives us the partition, but the original loss determines the values placed in the leaves.

Without this step, LAD would not get median leaf values, and Huber would not get its robust clipped update.

Transition: The implementation mirrors this separation directly.

## Slide 12 — From-Scratch Implementation

The implementation is organized around a small `Loss` interface.

Each loss supplies an initial prediction, a negative gradient, a terminal-region leaf update, and optionally an iteration-level state update. Huber uses that last method to update \(\delta_m\).

The boosting driver does not need to know whether it is running LS, LAD, or Huber. It simply calls these methods at the right places in the generic loop.

The tree code is also written from scratch: no scikit-learn, no XGBoost, no LightGBM, and no CatBoost. NumPy is used for array operations, but the boosting and tree logic are implemented directly.

Transition: The weak learner is a custom regression tree.

## Slide 13 — Custom Regression Tree

The tree uses best-first growth.

Instead of growing by a fixed depth, it repeatedly expands the leaf with the highest reduction in squared error until it reaches the desired number of leaves \(J\). This matches the complexity control used in the paper.

The split gain formula on the slide is the efficient form of the SSE reduction. It uses cumulative sums so all valid split points can be evaluated quickly after sorting a feature.

The `apply` and `set_leaf_values` API is important for boosting: after the tree is fit, the boosting driver can map rows to leaves and overwrite the raw least-squares leaf means with the loss-specific \(\gamma_{jm}\).

Transition: Now I will show whether the implementation reproduces the expected behavior.

## Slide 14 — Experiment 1: Clean Signal

The first experiment uses a Friedman-#1 style regression target with Gaussian noise.

All three methods use the same number of trees, learning rate, and number of leaves. The goal is to check basic convergence under clean, light-tailed noise.

The table shows that LS and Huber are essentially tied. Huber even has a slightly lower MSE and MAE here, though the difference is tiny. LAD is close in MAE but somewhat worse in MSE.

This matches the theory. Under Gaussian noise, squared error is efficient, Huber behaves nearly like squared error, and LAD pays a small efficiency cost because it ignores residual magnitude.

Transition: The real test for LAD and Huber is contamination.

## Slide 15 — Experiment 2: Robustness Under Contamination

In the second experiment, the target function is the same, but 10 percent of the training labels are perturbed by heavy-tailed shocks.

The test set is clean, so the metric asks whether the model recovered the underlying signal rather than the corrupted labels.

The results show the expected robustness pattern. LS collapses: its clean MSE jumps from about 2 to about 33. That happens because squared error gives extreme residuals enormous influence.

LAD recovers almost completely, with clean MSE around 3.49. Huber is much better than LS, with clean MSE around 11.28, but in this severe contamination setting it does not beat LAD.

The interpretation is that Huber's ranking depends on the threshold \(\delta\). It is a compromise between efficiency and robustness.

Transition: The scatter plot makes the same point visually.

## Slide 16 — Predicted vs True Under Contamination

This figure compares predicted values to the clean true signal.

For a well-calibrated model, points should lie close to the diagonal. LS drifts away from the diagonal because it has been pulled toward contaminated training labels.

LAD and Huber stay much closer to the diagonal. This visually confirms the numerical table from the previous slide.

The takeaway is that the robust losses are not just reducing a metric; they are preserving the underlying signal when the training labels contain outliers.

Transition: The last experiment studies shrinkage.

## Slide 17 — Experiment 3: Shrinkage Trade-Off

This experiment uses LS Boost on the clean target while varying the learning rate \(\nu\).

With \(\nu = 1.0\), the model reaches its best validation error quickly, after around 30 iterations, and then plateaus. With \(\nu = 0.1\), learning is slower but reaches a better final validation MSE. With \(\nu = 0.05\), learning is even slower and needs more trees to compete.

This reproduces one of the most important empirical lessons from the paper: smaller learning rates act as regularization, but they require more boosting iterations.

In practice, moderate shrinkage plus a larger number of trees is usually preferred.

Transition: I will now summarize what worked and what the limits were.

## Slide 18 — What Worked and What Did Not

Several pieces worked well.

The `Loss` interface matched the paper cleanly, which kept the boosting loop small and loss-agnostic. Best-first tree growth gave direct control over \(J\), the number of leaves. The unit tests were also useful because they checked the equations from the paper directly, especially the LAD and Huber leaf updates.

The experiments reproduced the qualitative claims of the paper: LS is the reality check, robust losses help under contamination, and shrinkage improves generalization.

The limits are also important. The CART splitter is correct but not optimized for large data. I did not implement stochastic subsampling or influence trimming, because the scope was regression TreeBoost. Huber's performance depends on the \(\delta\) quantile. Finally, LAD can be non-monotone within individual iterations because its pseudo-response uses only the sign of the residual.

Transition: I will close with the main lessons and the paper's legacy.

## Slide 19 — Results and Lasting Impact

The main result is that Friedman's function-space view is not just a nice explanation. It becomes a practical software design: once the loss exposes the right three operations, the whole boosting loop is reusable.

The terminal-region update is the most important implementation detail. It is what turns a generic tree fit into the correct LS, LAD, or Huber update.

The experiments also show the central trade-off. Robustness costs a little efficiency on clean data, but squared error can fail badly under contamination. So robust losses are valuable when outliers are plausible.

Historically, this paper created the conceptual foundation for modern gradient boosted tree libraries. XGBoost, LightGBM, and CatBoost add engineering, regularization, second-order approximations, and categorical handling, but the basic idea is Friedman's.

Transition: That is the end of the prepared talk.

## Slide 20 — Q&A

Thank you. I am happy to answer questions about the paper, the derivation, the implementation, or the experiments.

If helpful, the project includes the source code, tests, experiment scripts, generated figures, this slide deck, and the analysis writeup.

## Backup: Likely Questions

### Why fit the tree by least squares even for LAD or Huber?

Because the tree is approximating the negative gradient values at the training points. Least squares gives a simple way to fit a weak learner to those pseudo-responses. The original loss comes back in the terminal-region update.

### Why does LAD use medians in the leaves?

For absolute-error loss, the constant that minimizes the sum of absolute deviations inside a region is the median. So after the tree creates each region, the correct loss-specific update is the median residual in that region.

### Why did Huber not beat LAD in the contaminated experiment?

The contamination was severe and heavy-tailed. Huber clipped large residuals, so it was much better than LS, but its threshold still left outliers with nonzero influence. LAD is more aggressive because it uses only signs and median updates. With a different Huber quantile or milder contamination, Huber could rank closer to or above LAD.

### What is the difference between this implementation and XGBoost?

This implementation follows the core Friedman TreeBoost logic directly and prioritizes clarity. XGBoost adds second-order optimization, explicit regularization, optimized split finding, sparsity handling, parallelism, and production engineering.

### What would you improve next?

I would add stochastic subsampling, faster presorted or histogram-based split search, a grid for Huber's quantile parameter, and possibly quantile regression as another loss. Those would extend the same loss-agnostic architecture.
