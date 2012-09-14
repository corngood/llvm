//===--- DAGDeltaAlgorithm.h - A DAG Minimization Algorithm ----*- C++ -*--===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//===----------------------------------------------------------------------===//

#ifndef LLVM_ADT_DAGDELTAALGORITHM_H
#define LLVM_ADT_DAGDELTAALGORITHM_H

#include <vector>
#include <set>

namespace llvm {

/// DAGDeltaAlgorithm - Implements a "delta debugging" algorithm for minimizing
/// directed acyclic graphs using a predicate function.
///
/// The result of the algorithm is a subset of the input change set which is
/// guaranteed to satisfy the predicate, assuming that the input set did. For
/// well formed predicates, the result set is guaranteed to be such that
/// removing any single element not required by the dependencies on the other
/// elements would falsify the predicate.
///
/// The DAG should be used to represent dependencies in the changes which are
/// likely to hold across the predicate function. That is, for a particular
/// changeset S and predicate P:
///
///   P(S) => P(S union pred(S))
///
/// The minization algorithm uses this dependency information to attempt to
/// eagerly prune large subsets of changes. As with \see DeltaAlgorithm, the DAG
/// is not required to satisfy this property, but the algorithm will run
/// substantially fewer tests with appropriate dependencies. \see DeltaAlgorithm
/// for more information on the properties which the predicate function itself
/// should satisfy.
class DAGDeltaAlgorithm {
  virtual void anchor();
public:
  typedef unsigned change_ty;
  typedef std::pair<change_ty, change_ty> edge_ty;

  // FIXME: Use a decent data structure.
  typedef std::set<change_ty> changeset_ty;
  typedef std::vector<changeset_ty> changesetlist_ty;

public:
  virtual ~DAGDeltaAlgorithm() {}

  /// Run - Minimize the DAG formed by the \p Changes vertices and the
  /// \p Dependencies edges by executing \see ExecuteOneTest() on subsets of
  /// changes and returning the smallest set which still satisfies the test
  /// predicate and the input \p Dependencies.
  ///
  /// \param Changes The list of changes.
  ///
  /// \param Dependencies The list of dependencies amongst changes. For each
  /// (x,y) in \p Dependencies, both x and y must be in \p Changes. The
  /// minimization algorithm guarantees that for each tested changed set S,
  /// \f$ x \in S \f$ implies \f$ y \in S \f$. It is an error to have cyclic
  /// dependencies.
  changeset_ty Run(const changeset_ty &Changes,
                   const std::vector<edge_ty> &Dependencies);

  /// UpdatedSearchState - Callback used when the search state changes.
  virtual void UpdatedSearchState(const changeset_ty &Changes,
                                  const changesetlist_ty &Sets,
                                  const changeset_ty &Required) {}

  /// ExecuteOneTest - Execute a single test predicate on the change set \p S.
  virtual bool ExecuteOneTest(const changeset_ty &S) = 0;
};

} // end namespace llvm

#endif
