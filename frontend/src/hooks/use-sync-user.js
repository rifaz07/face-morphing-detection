'use client'
import { useEffect } from 'react'
import { useUser } from '@clerk/nextjs'

export function useSyncUser() {
  const { isSignedIn, isLoaded } = useUser()

  useEffect(() => {
    if (isLoaded && isSignedIn) {
      fetch('/api/auth/sync-user', { method: 'POST' })
        .catch(console.error)
    }
  }, [isLoaded, isSignedIn])
}
