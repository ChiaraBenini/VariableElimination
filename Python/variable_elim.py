"""
@Author: Joris van Vugt, Moira Berens, Leonieke van den Bulk

Class for the implementation of the variable elimination algorithm.
"""

import pandas as pd
import numpy as np
import random

class VariableElimination():
    def __init__(self, network):
        """
        Initialize the variable elimination algorithm with the specified network.
        """
        self.network = network

    def run(self, query, observed, heuristic):
        """
        Use the variable elimination algorithm to find out the probability
        distribution of the query variable given the observed variables.

        Input:
            query:      The query variable (string)
            observed:   A dictionary of the observed variables {variable: value}
            heuristic:  Either a heuristic string to determine the elimination ordering
                        or a list specifying the elimination ordering.

        Output: A pandas DataFrame holding the probability distribution
                for the query variable.
        """
        # === STEP 0: Apply evidence simplification! ===
        self.simplification(observed)

        # Determine elimination order (e.g., using a heuristic)
        elimination_order = self.elimination_order(heuristic, observed)
        # Remove the query from elimination ordering since we want to keep it.
        elimination_order = [var for var in elimination_order if var != query]
        #print("Elimination order:", elimination_order)

        # === STEP 1: Eliminate Variables ===
        factors = self.network.probabilities.copy()
        for var in elimination_order:
            # Find all factors that contain the variable.
            related_factors = [f for f in factors if var in factors[f].columns]
            if not related_factors:
                continue  # Nothing to eliminate for this variable.
            # Multiply all related factors.
            combined_factor = factors[related_factors[0]]
            for f in related_factors[1:]:
                combined_factor = self.multiplication(combined_factor, factors[f])
            # Sum out the variable.
            reduced_factor = self.sum_out(var, combined_factor)
            # Remove old factors and add the new reduced factor.
            for f in related_factors:
                del factors[f]
            factors[f"reduced_{var}"] = reduced_factor

        # === STEP 2: Multiply remaining factors ===
        final_factor = None
        for f in factors.values():
            if final_factor is None:
                final_factor = f
            else:
                final_factor = self.multiplication(final_factor, f)

        # === STEP 3: Remove extra variables (if any) that are not the query.
        # At this point, if evidence was applied correctly, evidence variables should be fixed.
        # However, sometimes evidence variables remain as columns.
        extra_vars = [col for col in final_factor.columns if col not in [query, 'prob']]
        for var in extra_vars:
            if var in observed:
                # For evidence variables, filter to the observed value and drop the column.
                final_factor = final_factor[final_factor[var] == observed[var]].drop(columns=[var])
            else:
                # Otherwise, sum them out.
                final_factor = self.sum_out(var, final_factor)

        # === STEP 4: Normalize the final factor ===
        final_factor['prob'] /= final_factor['prob'].sum()
        print("Final factor after normalization:")
        print(final_factor)
        return final_factor

    def simplification(self, observed):
        """
        Incorporate evidence by filtering each factor to only include rows consistent with the evidence.
        """
        for var, value in observed.items():
            #print(f"Applying evidence: {var} = {value}")
            for key in self.network.probabilities:
                df = self.network.probabilities[key]
                if var in df.columns:
                    # Filter rows where the variable equals the observed value and drop the variable.
                    df = df[df[var] == value].drop(columns=[var])
                    self.network.probabilities[key] = df

    def multiplication(self, factor1, factor2):
        """
        Multiply two factors (pandas DataFrames) together.
        If there are no common variables (other than 'prob'), add a dummy join key.
        """
        df1 = factor1 if isinstance(factor1, pd.DataFrame) else self.network.probabilities[factor1]
        df2 = factor2 if isinstance(factor2, pd.DataFrame) else self.network.probabilities[factor2]

        # Identify common variables (exclude 'prob').
        common_vars = list(set(df1.columns) & set(df2.columns))
        if "prob" in common_vars:
            common_vars.remove("prob")
        #print("Common variables:", common_vars)
        if not common_vars:
            df1 = df1.copy()
            df2 = df2.copy()
            df1['dummy'] = 1
            df2['dummy'] = 1
            common_vars = ['dummy']
            #print("No common variables. Added dummy join key.")

        merged_df = pd.merge(df1, df2, on=common_vars)
        merged_df['prob'] = merged_df['prob_x'] * merged_df['prob_y']
        merged_df.drop(columns=['prob_x', 'prob_y'], inplace=True)
        if 'dummy' in merged_df.columns:
            merged_df.drop(columns=['dummy'], inplace=True)
        return merged_df

    def sum_out(self, variable, factor):
        """
        Sum out (i.e. marginalize) a variable from a factor.
        """
        factor = factor if isinstance(factor, pd.DataFrame) else self.network.probabilities[factor]
        remaining_vars = [col for col in factor.columns if col not in [variable, 'prob']]
        new_factor = factor.groupby(remaining_vars)['prob'].sum().reset_index()
        return new_factor

    def elimination_order(self, heuristic, observed):
        """
        Compute an elimination ordering based on a heuristic.
        """
        remaining_variables = set(self.network.nodes) - set(observed.keys())
        elimination_order = []
        while remaining_variables:
            if heuristic == "random":
                var = random.choice(list(remaining_variables))
            elif heuristic == "min-size":
                var = min(remaining_variables, key=lambda var: len(self.network.probabilities[var]))
            elif heuristic == "least-incoming-arcs":
                var = min(remaining_variables, key=lambda var: len(self.network.parents[var]))
            elif heuristic == "outgoing-arcs-first":
                var = max(remaining_variables, key=lambda var: sum(1 for child in self.network.nodes if var in self.network.parents.get(child, [])))
            elif heuristic == "min-weight":
                var = min(remaining_variables, key=lambda var: len(self.network.probabilities[var]))
            elif heuristic == "fewest-factors":
                var = min(remaining_variables, key=lambda var: sum(1 for table in self.network.probabilities.values() if var in table.columns))
            else:
                var = random.choice(list(remaining_variables))
            elimination_order.append(var)
            remaining_variables.remove(var)
        return elimination_order
