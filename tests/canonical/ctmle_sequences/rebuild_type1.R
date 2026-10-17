# R `ctmle` 0.1.2 at 18de559: `ctmleGlmnet(ctmletype = 1)` taken apart, for the registered
# study `canonical-ctmle-sequences` and the candidate-sequence witness.  Line numbers cite
# `R/ctmle_glmnet.R` at that commit.  The study runner sources this file from
# `/fixture/ctmle_sequences/`, and the witness's `rebuild_check.R` from `/shared/`.  It defines
# functions only.
#
# `rebuild_type1` is the type-1 adapter (X16 plan, 3.4, S1).  `ctmleGlmnet` type 1 has no
# branch for a selected last penalty: `lambdas_remain` is empty at line 270, and the final
# step fails (quirk 7).  The function takes `ctmleGlmnet`'s formals (lines 86-95), runs its
# preamble (lines 97-112) verbatim, and rebuilds R's own construction and cross-validation
# (lines 213-230).  It returns the pieces `ctmleGlmnet` would have used, with
# `varDstar_origscale` as lines 346-347 compute it.
#
# `final_type1` is R's type-1 final step from a declared start (X16 plan, 3.4.2): the preamble,
# `stage2_glmnet(..., best_k = start)` as line 256 calls it, and the block of lines 248-327
# verbatim.  It returns what the replay reads: the explicit-lambda coefficients, the option
# losses, the final grid index and the estimate.  It needs a start before the last index; its
# caller, `final_columns`, checks that outside the stage.

ctmle_preamble <- function(Y, A, Q, alpha, fluctuation) {
      # ctmle_glmnet.R:97-112, verbatim.
      n <- length(Y)
      maptoYstar <- fluctuation=="logistic"
      Qbounds <- range(Y)
      QAW <- (1 - A) * Q[, 1] + A * Q[, 2]
      Q <- cbind(QAW, Q0W = Q[, 1], Q1W = Q[, 2])
      Q <- bound(Q, Qbounds)
      Ystar <- bound(Y, Qbounds)
      ab <- range(Ystar, na.rm = TRUE)
      Ystar[is.na(Ystar)] <- 0
      Ystar <- (Ystar - ab[1])/diff(ab)
      Q <- (Q - ab[1])/diff(ab)
      Qbounds <- c(alpha, 1 - alpha)
      Q <- bound(Q, Qbounds)
      Q <- qlogis(Q)
      family.stage2 <-  "binomial"
      Qinit<-list(Q=Q, Y=Ystar, Qbounds=Qbounds, ab=ab, family=family.stage2)
      list(Qinit = Qinit, n = n, maptoYstar = maptoYstar)
}

rebuild_type1 <- function(Y, A, W, Wg = W,
                          Q, lambdas = NULL,
                          ctmletype,
                          V=5, folds = NULL,
                          alpha=.995, family= "gaussian",
                          gbound=.025, like_type = "RSS",
                          fluctuation="logistic", verbose=FALSE,
                          detailed=FALSE, PEN=FALSE,
                          g1W = NULL, g1WPrev = NULL,
                          stopFactor=10^6) {
      start <- ctmle_preamble(Y, A, Q, alpha, fluctuation)
      Qinit <- start$Qinit
      maptoYstar <- start$maptoYstar

      # ctmle_glmnet.R:213-219 and 224-230, with the internal functions named.
      candidates <- ctmle:::stage2_glmnet(Y=Qinit$Y, X=data.frame(A,Wg), Q=Qinit$Q,
                                          lambdas = lambdas,ctmletype=ctmletype,
                                          family=Qinit$family,
                                          Qbounds=Qinit$Qbounds,ab=Qinit$ab,
                                          training_set=rep(TRUE,length(Y)),
                                          like_type=like_type,gbound=gbound, verbose=verbose,
                                          stopFactor=stopFactor)

      cv.res <- ctmle:::cv_glmnet(Y=Qinit$Y,X=data.frame(A,Wg),
                                  est.all=candidates$results.all$est,
                                  Q=Qinit$Q,lambdas=lambdas,ctmletype=ctmletype,
                                  family=Qinit$family,
                                  Qbounds=Qinit$Qbounds, ab=Qinit$ab,
                                  like_type=like_type,gbound=gbound,
                                  verbose=verbose, PEN=PEN , V=V, folds = folds)

      # ctmle_glmnet.R:346-347, verbatim.
      candidates$results.all$varDstar_origscale <- candidates$results.all$varDstar *
            ifelse(maptoYstar, (diff(Qinit$ab))^2, 1)

      list(candidates = candidates, cv.res = cv.res, n = start$n)
}

final_type1 <- function(Y, A, W, Wg = W, Q, lambdas, start_k,
                        alpha=.995, gbound=.025, like_type = "RSS",
                        fluctuation="logistic", verbose=FALSE,
                        stopFactor=10^6) {
      start <- ctmle_preamble(Y, A, Q, alpha, fluctuation)
      Qinit <- start$Qinit
      ctmletype <- 1
      # `cv.res$best_k` is the only field of `cv.res` that lines 248-327 read.
      cv.res <- list(best_k = start_k)
      # `bound` comes from `tmle`, which `ctmle` attaches.
      stage2_glmnet <- ctmle:::stage2_glmnet

      # ctmle_glmnet.R:248-327, verbatim.
                  Y=Qinit$Y
                  X=data.frame(A,Wg)
                  family=Qinit$family
                  Qbounds=Qinit$Qbounds
                  ab=Qinit$ab
                  gbound=gbound
                  # Cheng: This extra step guarantee the critival equation is solved!
                  # Modify the stage2_glmnet function to return the corresponding Qstar
                  candidates_detail <- stage2_glmnet(Y=Qinit$Y, X=data.frame(A,Wg), Q=Qinit$Q,
                                                     lambdas=lambdas,ctmletype=ctmletype,
                                                     family=Qinit$family,
                                                     Qbounds=Qinit$Qbounds,ab=Qinit$ab,
                                                     training_set=rep(TRUE,length(Y)),
                                                     like_type=like_type,gbound=gbound, verbose=verbose,
                                                     stopFactor=stopFactor,
                                                     best_k = cv.res$best_k)

                  Qstar_best <- candidates_detail$results.all$Qstar_best
                  # Print the best result before solving critical equation!
                  # mean(Qstar_best[,3] - Qstar_best[,2]) * diff(ab)
                  #------------------------------------------------

                  lambdas_remain <- lambdas[lambdas < lambdas[cv.res$best_k]]

                  f1 <- eval(paste("A ~ ", paste(paste(names(X[,-1]), sep=''), collapse=" + ")))
                  f2 <- as.formula(f1)
                  lassox <- model.matrix(f2, X)[,-1]
                  lassoy <-as.factor(X[,1])

                  glmnet_remain <- glmnet(x=lassox,y=lassoy,family="binomial", lambda= lambdas_remain)
                  gn_remain <- predict(glmnet_remain, newx = lassox,
                                       type="response")

                  gn_remain <-bound(gn_remain, c(gbound, 1- gbound))

                  finalTMLE <- function(g_tmp, detail = FALSE){

                        g1W.total <- bound(g_tmp, c(gbound, 1-gbound))
                        g0W.total <- 1 - bound(g_tmp, c(gbound, 1-gbound))

                        H1W <- X[,1]/g1W.total
                        H0W <- (1 - X[,1])/g0W.total

                        suppressWarnings(glm_tmp <- glm(Y ~ -1 + offset(qlogis(Qstar_best[, "QAW"]))
                                                        + H0W+H1W, family = 'binomial'))
                        epsilon<- coef(glm_tmp)

                        Qstar_tmp <- qlogis(Qstar_best) + cbind((epsilon[1] * H0W+ epsilon[2] * H1W),
                                                                epsilon[1]/g0W.total, epsilon[2]/g1W.total)


                        Qstar_tmp <- plogis(Qstar_tmp)


                        mu1 <- mean(Qstar_tmp[,"Q1W"])
                        mu0 <- mean(Qstar_tmp[,"Q0W"])
                        est  <- (mean(Qstar_tmp[,"Q1W"]) - mean(Qstar_tmp[,"Q0W"]))*diff(ab)

                        Dstar <-  (A/g1W.total - (1 - A)/(1-g1W.total))  * (Y - Qstar_tmp[, "QAW"]) +
                              Qstar_tmp[, "Q1W"] -Qstar_tmp[, "Q0W"] -
                              (mean(Qstar_tmp[,"Q1W"]) - mean(Qstar_tmp[,"Q0W"]))
                        # mean(Dstar)
                        # browser()
                        if(!detail){
                              return(rbind(est
                                           , -logLik(glm_tmp)[1]))
                        }else{

                              var.psi_tmp <- var(Dstar)/length(Y) *(diff(ab)^2)
                              CI <- est + c(-1.96,1.96) * sqrt(var.psi_tmp)
                              CI
                              return(list(est, CI, var.psi = var.psi_tmp))
                        }
                  }

                  tmle_loss <- t(apply(gn_remain, 2, finalTMLE))[,2]
                  # !! Find the minimizer of the empirical loss
                  ind_final <- which.min(tmle_loss)

                  res <- finalTMLE(gn_remain[, ind_final], detail = TRUE)
      # End of ctmle_glmnet.R:248-327.

      list(
            start = as.integer(start_k),
            coef = as.matrix(coef(glmnet_remain)),
            response = predict(glmnet_remain, newx = lassox, type = "response"),
            lassox = lassox,
            loss = as.numeric(tmle_loss),
            index = as.integer(start_k + ind_final),
            est = as.numeric(res[[1]]),
            var.psi = as.numeric(res[[3]])
      )
}
