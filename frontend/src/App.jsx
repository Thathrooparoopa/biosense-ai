function App() {
  return (
    <main className="flex min-h-screen items-center justify-center bg-[#070B14] px-6">
      <section className="text-center">
        <p className="mb-3 text-sm font-medium uppercase tracking-[0.3em] text-indigo-400">
          BioSense AI
        </p>

        <h1 className="text-4xl font-bold tracking-tight text-white sm:text-5xl">
          Virtual Electronic Nose
        </h1>

        <p className="mx-auto mt-4 max-w-xl text-slate-400">
          AI-assisted research platform for analyzing breath, VOC, and
          electronic-nose sensor patterns.
        </p>

        <div className="mt-8 inline-flex rounded-full border border-indigo-500/30 bg-indigo-500/10 px-5 py-2 text-sm text-indigo-300">
          Frontend setup successful
        </div>
      </section>
    </main>
  )
}

export default App