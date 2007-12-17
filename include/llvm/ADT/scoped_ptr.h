//===- llvm/ADT/scoped_ptr.h - basic smart pointer --------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines the scoped_ptr smart pointer: scoped_ptr mimics a built-in
// pointer except that it guarantees deletion of the object pointed to, either
// on destruction of the scoped_ptr or via an explicit reset(). scoped_ptr is a
// simple solution for simple needs.
//
//===----------------------------------------------------------------------===//
//
//  (C) Copyright Greg Colvin and Beman Dawes 1998, 1999.
//  Copyright (c) 2001, 2002 Peter Dimov
//
//  Distributed under the Boost Software License, Version 1.0. (See
//  accompanying file llvm/docs/BOOST_LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt )
//
//  http://www.boost.org/libs/smart_ptr/scoped_ptr.htm
//
#ifndef LLVM_SCOPED_PTR_H_INCLUDED
#define LLVM_SCOPED_PTR_H_INCLUDED

#include <cassert>

namespace llvm {

// verify that types are complete for increased safety
template<class T>
inline void checked_delete(T * x) {
  // intentionally complex - simplification causes warnings in some compilers.
  typedef char type_must_be_complete[sizeof(T) ? 1 : -1];
  (void)sizeof(type_must_be_complete);
  delete x;
}

/// scoped_ptr mimics a built-in pointer except that it guarantees deletion
/// of the object pointed to, either on destruction of the scoped_ptr or via
/// an explicit reset(). scoped_ptr is a simple solution for simple needs;
/// use shared_ptr or std::auto_ptr if your needs are more complex.
template<class T> 
class scoped_ptr {// noncopyable
  T *ptr;
  scoped_ptr(scoped_ptr const &);             // DO NOT IMPLEMENT
  scoped_ptr & operator=(scoped_ptr const &); // DO NOT IMPLEMENT
  typedef scoped_ptr<T> this_type;
public:
  typedef T element_type;

  explicit scoped_ptr(T * p = 0): ptr(p) {} // never throws

  ~scoped_ptr() { // never throws
    llvm::checked_delete(ptr);
  }

  /// reset - Change the current pointee to the specified pointer.  Note that
  /// calling this with any pointer (including a null pointer) deletes the
  /// current pointer.
  void reset(T *p = 0) { 
    // catch self-reset errors
    assert((p == 0 || p != ptr) && "scoped_ptr: self-reset error");
    T *tmp = ptr;
    ptr = p;
    delete tmp;
  }

  /// take - Reset the scoped pointer to null and return its pointer.  This does
  /// not delete the pointer before returning it.
  T *take() { 
    T *P = ptr;
    ptr = 0;
    return P;
  }
  
  T& operator*() const {
    assert(ptr != 0 && "scoped_ptr: Trying to dereference a null pointer");
    return *ptr;
  }

  T* operator->() const {
    assert(ptr != 0 && "scoped_ptr: Trying to dereference a null pointer");
    return ptr;
  }

  T* get() const {
    return ptr;
  }

  // implicit conversion to "bool"
  typedef T * this_type::*unspecified_bool_type;

  operator unspecified_bool_type() const {// never throws
    return ptr == 0? 0: &this_type::ptr;
  }

  bool operator!() const { // never throws
    return ptr == 0;
  }

  void swap(scoped_ptr &b) {// never throws
    T * tmp = b.ptr;
    b.ptr = ptr;
    ptr = tmp;
  }
};

template<class T> inline void swap(scoped_ptr<T> &a, scoped_ptr<T> &b) {
  // never throws
  a.swap(b);
}

// get_pointer(p) is a generic way to say p.get()
template<class T> inline T * get_pointer(scoped_ptr<T> const &p) {
  return p.get();
}

} // namespace llvm

#endif // #ifndef LLVM_SCOPED_PTR_HPP_INCLUDED
