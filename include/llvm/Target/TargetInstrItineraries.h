//===-- llvm/Target/TargetInstrItineraries.h - Scheduling -------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file describes the structures used for instruction itineraries and
// states.  This is used by schedulers to determine instruction states and
// latencies.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TARGET_TARGETINSTRITINERARIES_H
#define LLVM_TARGET_TARGETINSTRITINERARIES_H

namespace llvm {

//===----------------------------------------------------------------------===//
/// Instruction stage - These values represent a step in the execution of an
/// instruction.  The latency represents the number of discrete time slots used
/// need to complete the stage.  Units represent the choice of functional units
/// that can be used to complete the stage.  Eg. IntUnit1, IntUnit2.
///
struct InstrStage {
  unsigned Cycles;  ///< Length of stage in machine cycles
  unsigned Units;   ///< Choice of functional units
};


//===----------------------------------------------------------------------===//
/// Instruction itinerary - An itinerary represents a sequential series of steps
/// required to complete an instruction.  Itineraries are represented as
/// sequences of instruction stages.
///
struct InstrItinerary {
  unsigned First;    ///< Index of first stage in itinerary
  unsigned Last;     ///< Index of last + 1 stage in itinerary
};



//===----------------------------------------------------------------------===//
/// Instruction itinerary Data - Itinerary data supplied by a subtarget to be
/// used by a target.
///
struct InstrItineraryData {
  const InstrStage     *Stages;         ///< Array of stages selected
  const InstrItinerary *Itineratries;   ///< Array of itineraries selected

  /// Ctors.
  ///
  InstrItineraryData() : Stages(0), Itineratries(0) {}
  InstrItineraryData(const InstrStage *S, const InstrItinerary *I)
    : Stages(S), Itineratries(I) {}
  
  /// isEmpty - Returns true if there are no itineraries.
  ///
  bool isEmpty() const { return Itineratries == 0; }
  
  /// begin - Return the first stage of the itinerary.
  /// 
  const InstrStage *begin(unsigned ItinClassIndx) const {
    unsigned StageIdx = Itineratries[ItinClassIndx].First;
    return Stages + StageIdx;
  }

  /// end - Return the last+1 stage of the itinerary.
  /// 
  const InstrStage *end(unsigned ItinClassIndx) const {
    unsigned StageIdx = Itineratries[ItinClassIndx].Last;
    return Stages + StageIdx;
  }
};


} // End llvm namespace

#endif
