% ================= EXERCICE 6 =================
\section*{Exercice 6 : Évaluation et Comparaison des Méthodes de Recherche}

\subsection*{Q1. Comparaison entre Grid Search et Random Search}

\textbf{Résultats obtenus} :
\begin{table}[H]
    \centering
    \begin{tabular}{lc}
        \toprule
        Méthode & Validation MSE \\
        \midrule
        Best Grid Search & 0.2656 \\
        Best Random Search & 0.3044 \\
        2nd Random Search & 0.3389 \\
        3rd Random Search & 0.3432 \\
        \bottomrule
    \end{tabular}
    \caption{Comparaison Grid Search vs Random Search}
\end{table}

\textbf{Meilleure configuration Grid Search} :
\begin{itemize}
    \item Architecture : [128, 64, 32]
    \item Activation : Leaky ReLU
    \item Dropout : 0.1
    \item Validation MSE : 0.2656
\end{itemize}

\subsection*{Conclusion de l'exercice 6}
Le Grid Search a permis d'obtenir le meilleur modèle avec une Validation MSE de 0.2656.

\newpage