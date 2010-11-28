//===---- ADT/IntervalMapTest.cpp - IntervalMap unit tests ------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "llvm/ADT/IntervalMap.h"
#include "gtest/gtest.h"

using namespace llvm;

namespace {

typedef IntervalMap<unsigned, unsigned> UUMap;
typedef IntervalMap<unsigned, unsigned, 4> UU4Map;

// Empty map tests
TEST(IntervalMapTest, EmptyMap) {
  UUMap::Allocator allocator;
  UUMap map(allocator);
  EXPECT_TRUE(map.empty());

  // Lookup on empty map.
  EXPECT_EQ(0u, map.lookup(0));
  EXPECT_EQ(7u, map.lookup(0, 7));
  EXPECT_EQ(0u, map.lookup(~0u-1));
  EXPECT_EQ(7u, map.lookup(~0u-1, 7));

  // Iterators.
  EXPECT_TRUE(map.begin() == map.begin());
  EXPECT_TRUE(map.begin() == map.end());
  EXPECT_TRUE(map.end() == map.end());
  EXPECT_FALSE(map.begin() != map.begin());
  EXPECT_FALSE(map.begin() != map.end());
  EXPECT_FALSE(map.end() != map.end());
  EXPECT_FALSE(map.begin().valid());
  EXPECT_FALSE(map.end().valid());
  UUMap::iterator I = map.begin();
  EXPECT_FALSE(I.valid());
  EXPECT_TRUE(I == map.end());
}

// Single entry map tests
TEST(IntervalMapTest, SingleEntryMap) {
  UUMap::Allocator allocator;
  UUMap map(allocator);
  map.insert(100, 150, 1);
  EXPECT_FALSE(map.empty());

  // Lookup around interval.
  EXPECT_EQ(0u, map.lookup(0));
  EXPECT_EQ(0u, map.lookup(99));
  EXPECT_EQ(1u, map.lookup(100));
  EXPECT_EQ(1u, map.lookup(101));
  EXPECT_EQ(1u, map.lookup(125));
  EXPECT_EQ(1u, map.lookup(149));
  EXPECT_EQ(1u, map.lookup(150));
  EXPECT_EQ(0u, map.lookup(151));
  EXPECT_EQ(0u, map.lookup(200));
  EXPECT_EQ(0u, map.lookup(~0u-1));

  // Iterators.
  EXPECT_TRUE(map.begin() == map.begin());
  EXPECT_FALSE(map.begin() == map.end());
  EXPECT_TRUE(map.end() == map.end());
  EXPECT_TRUE(map.begin().valid());
  EXPECT_FALSE(map.end().valid());

  // Iter deref.
  UUMap::iterator I = map.begin();
  ASSERT_TRUE(I.valid());
  EXPECT_EQ(100u, I.start());
  EXPECT_EQ(150u, I.stop());
  EXPECT_EQ(1u, I.value());

  // Preincrement.
  ++I;
  EXPECT_FALSE(I.valid());
  EXPECT_FALSE(I == map.begin());
  EXPECT_TRUE(I == map.end());

  // PreDecrement.
  --I;
  ASSERT_TRUE(I.valid());
  EXPECT_EQ(100u, I.start());
  EXPECT_EQ(150u, I.stop());
  EXPECT_EQ(1u, I.value());
  EXPECT_TRUE(I == map.begin());
  EXPECT_FALSE(I == map.end());

  I.erase();
  EXPECT_TRUE(map.empty());
  EXPECT_EQ(0, std::distance(map.begin(), map.end()));
}

// Flat coalescing tests.
TEST(IntervalMapTest, RootCoalescing) {
  UUMap::Allocator allocator;
  UUMap map(allocator);
  map.insert(100, 150, 1);

  // Coalesce from the left.
  map.insert(90, 99, 1);
  EXPECT_EQ(1, std::distance(map.begin(), map.end()));
  EXPECT_EQ(90u, map.start());
  EXPECT_EQ(150u, map.stop());

  // Overlap left.
  map.insert(80, 100, 1);
  EXPECT_EQ(1, std::distance(map.begin(), map.end()));
  EXPECT_EQ(80u, map.start());
  EXPECT_EQ(150u, map.stop());

  // Inside.
  map.insert(100, 130, 1);
  EXPECT_EQ(1, std::distance(map.begin(), map.end()));
  EXPECT_EQ(80u, map.start());
  EXPECT_EQ(150u, map.stop());

  // Overlap both.
  map.insert(70, 160, 1);
  EXPECT_EQ(1, std::distance(map.begin(), map.end()));
  EXPECT_EQ(70u, map.start());
  EXPECT_EQ(160u, map.stop());

  // Overlap right.
  map.insert(80, 170, 1);
  EXPECT_EQ(1, std::distance(map.begin(), map.end()));
  EXPECT_EQ(70u, map.start());
  EXPECT_EQ(170u, map.stop());

  // Coalesce from the right.
  map.insert(170, 200, 1);
  EXPECT_EQ(1, std::distance(map.begin(), map.end()));
  EXPECT_EQ(70u, map.start());
  EXPECT_EQ(200u, map.stop());

  // Non-coalesce from the left.
  map.insert(60, 69, 2);
  EXPECT_EQ(2, std::distance(map.begin(), map.end()));
  EXPECT_EQ(60u, map.start());
  EXPECT_EQ(200u, map.stop());
  EXPECT_EQ(2u, map.lookup(69));
  EXPECT_EQ(1u, map.lookup(70));

  UUMap::iterator I = map.begin();
  EXPECT_EQ(60u, I.start());
  EXPECT_EQ(69u, I.stop());
  EXPECT_EQ(2u, I.value());
  ++I;
  EXPECT_EQ(70u, I.start());
  EXPECT_EQ(200u, I.stop());
  EXPECT_EQ(1u, I.value());
  ++I;
  EXPECT_FALSE(I.valid());

  // Non-coalesce from the right.
  map.insert(201, 210, 2);
  EXPECT_EQ(3, std::distance(map.begin(), map.end()));
  EXPECT_EQ(60u, map.start());
  EXPECT_EQ(210u, map.stop());
  EXPECT_EQ(2u, map.lookup(201));
  EXPECT_EQ(1u, map.lookup(200));

  // Erase from the left.
  map.begin().erase();
  EXPECT_EQ(2, std::distance(map.begin(), map.end()));
  EXPECT_EQ(70u, map.start());
  EXPECT_EQ(210u, map.stop());

  // Erase from the right.
  (--map.end()).erase();
  EXPECT_EQ(1, std::distance(map.begin(), map.end()));
  EXPECT_EQ(70u, map.start());
  EXPECT_EQ(200u, map.stop());
}

// Flat multi-coalescing tests.
TEST(IntervalMapTest, RootMultiCoalescing) {
  UUMap::Allocator allocator;
  UUMap map(allocator);
  map.insert(140, 150, 1);
  map.insert(160, 170, 1);
  map.insert(100, 110, 1);
  map.insert(120, 130, 1);
  EXPECT_EQ(4, std::distance(map.begin(), map.end()));
  EXPECT_EQ(100u, map.start());
  EXPECT_EQ(170u, map.stop());

  // Verify inserts.
  UUMap::iterator I = map.begin();
  EXPECT_EQ(100u, I.start());
  EXPECT_EQ(110u, I.stop());
  ++I;
  EXPECT_EQ(120u, I.start());
  EXPECT_EQ(130u, I.stop());
  ++I;
  EXPECT_EQ(140u, I.start());
  EXPECT_EQ(150u, I.stop());
  ++I;
  EXPECT_EQ(160u, I.start());
  EXPECT_EQ(170u, I.stop());
  ++I;
  EXPECT_FALSE(I.valid());

  // Test advanceTo on flat tree.
  I = map.begin();
  I.advanceTo(135);
  ASSERT_TRUE(I.valid());
  EXPECT_EQ(140u, I.start());
  EXPECT_EQ(150u, I.stop());

  I.advanceTo(145);
  ASSERT_TRUE(I.valid());
  EXPECT_EQ(140u, I.start());
  EXPECT_EQ(150u, I.stop());

  // Coalesce left with followers.
  // [100;110] [120;130] [140;150] [160;170]
  map.insert(111, 115, 1);
  I = map.begin();
  ASSERT_TRUE(I.valid());
  EXPECT_EQ(100u, I.start());
  EXPECT_EQ(115u, I.stop());
  ++I;
  ASSERT_TRUE(I.valid());
  EXPECT_EQ(120u, I.start());
  EXPECT_EQ(130u, I.stop());
  ++I;
  ASSERT_TRUE(I.valid());
  EXPECT_EQ(140u, I.start());
  EXPECT_EQ(150u, I.stop());
  ++I;
  ASSERT_TRUE(I.valid());
  EXPECT_EQ(160u, I.start());
  EXPECT_EQ(170u, I.stop());
  ++I;
  EXPECT_FALSE(I.valid());

  // Coalesce right with followers.
  // [100;115] [120;130] [140;150] [160;170]
  map.insert(135, 139, 1);
  I = map.begin();
  ASSERT_TRUE(I.valid());
  EXPECT_EQ(100u, I.start());
  EXPECT_EQ(115u, I.stop());
  ++I;
  ASSERT_TRUE(I.valid());
  EXPECT_EQ(120u, I.start());
  EXPECT_EQ(130u, I.stop());
  ++I;
  ASSERT_TRUE(I.valid());
  EXPECT_EQ(135u, I.start());
  EXPECT_EQ(150u, I.stop());
  ++I;
  ASSERT_TRUE(I.valid());
  EXPECT_EQ(160u, I.start());
  EXPECT_EQ(170u, I.stop());
  ++I;
  EXPECT_FALSE(I.valid());

  // Coalesce left and right with followers.
  // [100;115] [120;130] [135;150] [160;170]
  map.insert(131, 134, 1);
  I = map.begin();
  ASSERT_TRUE(I.valid());
  EXPECT_EQ(100u, I.start());
  EXPECT_EQ(115u, I.stop());
  ++I;
  ASSERT_TRUE(I.valid());
  EXPECT_EQ(120u, I.start());
  EXPECT_EQ(150u, I.stop());
  ++I;
  ASSERT_TRUE(I.valid());
  EXPECT_EQ(160u, I.start());
  EXPECT_EQ(170u, I.stop());
  ++I;
  EXPECT_FALSE(I.valid());

  // Coalesce multiple with overlap right.
  // [100;115] [120;150] [160;170]
  map.insert(116, 165, 1);
  I = map.begin();
  ASSERT_TRUE(I.valid());
  EXPECT_EQ(100u, I.start());
  EXPECT_EQ(170u, I.stop());
  ++I;
  EXPECT_FALSE(I.valid());

  // Coalesce multiple with overlap left
  // [100;170]
  map.insert(180, 190, 1);
  map.insert(200, 210, 1);
  map.insert(220, 230, 1);
  // [100;170] [180;190] [200;210] [220;230]
  map.insert(160, 199, 1);
  I = map.begin();
  ASSERT_TRUE(I.valid());
  EXPECT_EQ(100u, I.start());
  EXPECT_EQ(210u, I.stop());
  ++I;
  ASSERT_TRUE(I.valid());
  EXPECT_EQ(220u, I.start());
  EXPECT_EQ(230u, I.stop());
  ++I;
  EXPECT_FALSE(I.valid());

  // Overwrite 2 from gap to gap.
  // [100;210] [220;230]
  map.insert(50, 250, 1);
  I = map.begin();
  ASSERT_TRUE(I.valid());
  EXPECT_EQ(50u, I.start());
  EXPECT_EQ(250u, I.stop());
  ++I;
  EXPECT_FALSE(I.valid());

  // Coalesce at end of full root.
  // [50;250]
  map.insert(260, 270, 1);
  map.insert(280, 290, 1);
  map.insert(300, 310, 1);
  // [50;250] [260;270] [280;290] [300;310]
  map.insert(311, 320, 1);
  I = map.begin();
  ASSERT_TRUE(I.valid());
  EXPECT_EQ(50u, I.start());
  EXPECT_EQ(250u, I.stop());
  ++I;
  ASSERT_TRUE(I.valid());
  EXPECT_EQ(260u, I.start());
  EXPECT_EQ(270u, I.stop());
  ++I;
  ASSERT_TRUE(I.valid());
  EXPECT_EQ(280u, I.start());
  EXPECT_EQ(290u, I.stop());
  ++I;
  ASSERT_TRUE(I.valid());
  EXPECT_EQ(300u, I.start());
  EXPECT_EQ(320u, I.stop());
  ++I;
  EXPECT_FALSE(I.valid());

  // Test clear() on non-branched map.
  map.clear();
  EXPECT_TRUE(map.empty());
  EXPECT_TRUE(map.begin() == map.end());
}

// Branched, non-coalescing tests.
TEST(IntervalMapTest, Branched) {
  UUMap::Allocator allocator;
  UUMap map(allocator);

  // Insert enough intervals to force a branched tree.
  // This creates 9 leaf nodes with 11 elements each, tree height = 1.
  for (unsigned i = 1; i < 100; ++i) {
    map.insert(10*i, 10*i+5, i);
    EXPECT_EQ(10u, map.start());
    EXPECT_EQ(10*i+5, map.stop());
  }

  // Tree limits.
  EXPECT_FALSE(map.empty());
  EXPECT_EQ(10u, map.start());
  EXPECT_EQ(995u, map.stop());

  // Tree lookup.
  for (unsigned i = 1; i < 100; ++i) {
    EXPECT_EQ(0u, map.lookup(10*i-1));
    EXPECT_EQ(i, map.lookup(10*i));
    EXPECT_EQ(i, map.lookup(10*i+5));
    EXPECT_EQ(0u, map.lookup(10*i+6));
  }

  // Forward iteration.
  UUMap::iterator I = map.begin();
  for (unsigned i = 1; i < 100; ++i) {
    ASSERT_TRUE(I.valid());
    EXPECT_EQ(10*i, I.start());
    EXPECT_EQ(10*i+5, I.stop());
    EXPECT_EQ(i, *I);
    ++I;
  }
  EXPECT_FALSE(I.valid());
  EXPECT_TRUE(I == map.end());

  // Backwards iteration.
  for (unsigned i = 99; i; --i) {
    --I;
    ASSERT_TRUE(I.valid());
    EXPECT_EQ(10*i, I.start());
    EXPECT_EQ(10*i+5, I.stop());
    EXPECT_EQ(i, *I);
  }
  EXPECT_TRUE(I == map.begin());

  // Test advanceTo in same node.
  I.advanceTo(20);
  ASSERT_TRUE(I.valid());
  EXPECT_EQ(20u, I.start());
  EXPECT_EQ(25u, I.stop());

  // advanceTo another node.
  I.advanceTo(200);
  ASSERT_TRUE(I.valid());
  EXPECT_EQ(200u, I.start());
  EXPECT_EQ(205u, I.stop());

  // Erase from the front.
  I = map.begin();
  for (unsigned i = 0; i != 20; ++i) {
    I.erase();
    EXPECT_TRUE(I == map.begin());
    EXPECT_FALSE(map.empty());
    EXPECT_EQ(I.start(), map.start());
    EXPECT_EQ(995u, map.stop());
  }

  // Test clear() on branched map.
  map.clear();
  EXPECT_TRUE(map.empty());
  EXPECT_TRUE(map.begin() == map.end());
}

// Branched, high, non-coalescing tests.
TEST(IntervalMapTest, Branched2) {
  UU4Map::Allocator allocator;
  UU4Map map(allocator);

  // Insert enough intervals to force a height >= 2 tree.
  for (unsigned i = 1; i < 1000; ++i)
    map.insert(10*i, 10*i+5, i);

  // Tree limits.
  EXPECT_FALSE(map.empty());
  EXPECT_EQ(10u, map.start());
  EXPECT_EQ(9995u, map.stop());

  // Tree lookup.
  for (unsigned i = 1; i < 1000; ++i) {
    EXPECT_EQ(0u, map.lookup(10*i-1));
    EXPECT_EQ(i, map.lookup(10*i));
    EXPECT_EQ(i, map.lookup(10*i+5));
    EXPECT_EQ(0u, map.lookup(10*i+6));
  }

  // Forward iteration.
  UU4Map::iterator I = map.begin();
  for (unsigned i = 1; i < 1000; ++i) {
    ASSERT_TRUE(I.valid());
    EXPECT_EQ(10*i, I.start());
    EXPECT_EQ(10*i+5, I.stop());
    EXPECT_EQ(i, *I);
    ++I;
  }
  EXPECT_FALSE(I.valid());
  EXPECT_TRUE(I == map.end());

  // Backwards iteration.
  for (unsigned i = 999; i; --i) {
    --I;
    ASSERT_TRUE(I.valid());
    EXPECT_EQ(10*i, I.start());
    EXPECT_EQ(10*i+5, I.stop());
    EXPECT_EQ(i, *I);
  }
  EXPECT_TRUE(I == map.begin());

  // Test advanceTo in same node.
  I.advanceTo(20);
  ASSERT_TRUE(I.valid());
  EXPECT_EQ(20u, I.start());
  EXPECT_EQ(25u, I.stop());

  // advanceTo sibling leaf node.
  I.advanceTo(200);
  ASSERT_TRUE(I.valid());
  EXPECT_EQ(200u, I.start());
  EXPECT_EQ(205u, I.stop());

  // advanceTo further.
  I.advanceTo(2000);
  ASSERT_TRUE(I.valid());
  EXPECT_EQ(2000u, I.start());
  EXPECT_EQ(2005u, I.stop());

  // Test clear() on branched map.
  map.clear();
  EXPECT_TRUE(map.empty());
  EXPECT_TRUE(map.begin() == map.end());
}

// Random insertions, coalescing to a single interval.
TEST(IntervalMapTest, RandomCoalescing) {
  UU4Map::Allocator allocator;
  UU4Map map(allocator);

  // This is a poor PRNG with maximal period:
  // x_n = 5 x_{n-1} + 1 mod 2^N

  unsigned x = 100;
  for (unsigned i = 0; i != 4096; ++i) {
    map.insert(10*x, 10*x+9, 1);
    EXPECT_GE(10*x, map.start());
    EXPECT_LE(10*x+9, map.stop());
    x = (5*x+1)%4096;
  }

  // Map should be fully coalesced after that exercise.
  EXPECT_FALSE(map.empty());
  EXPECT_EQ(0u, map.start());
  EXPECT_EQ(40959u, map.stop());
  EXPECT_EQ(1, std::distance(map.begin(), map.end()));

}

} // namespace
